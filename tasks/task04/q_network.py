#!/usr/bin/env python3
import argparse
import collections
import os
import sys
import zipfile

# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()

# BATCH
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")

parser.add_argument("--epsilon_decay", default=0.999, type=float, help="Exploration factor.")
# EPSIOLON
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=20*10000, type=int, help="Training episodes.")

# GAMMA
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")

# UPDATE Q (step, not episode)
parser.add_argument("--target_update_freq", default=20, type=int, help="Target update frequency.")

parser.add_argument("--hidden_layer_size", default=16, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.003, type=float, help="Learning rate.")

# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=1000, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class MeanValueSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        # [q, a]
        # [q, a]
        # 0 if not the action
        # (qtrue - qfalse)^2
        #
        q_true = tf.expand_dims(tf.gather_nd(tf.transpose(y_true), [0]), axis=1)
        indexes = tf.cast(tf.expand_dims(tf.gather_nd(tf.transpose(y_true), [1]), axis=1), dtype=tf.int32)

        # create the row index with tf.range
        row_idx = tf.reshape(tf.range(indexes.shape[0]), (-1, 1))
        # stack with column index
        idx = tf.stack([row_idx, indexes], axis=-1)
        # extract the elements with gather_nd
        values = tf.cast(tf.gather_nd(y_pred, idx), dtype=tf.float64)

        # tf.print(y_true)
        # tf.print(y_pred)

        res = tf.math.pow(q_true - values, 2)
        # tf.print(res)
        res = tf.reshape(res, [-1])
        # nyni 2 stejne sloupecky
        # res = tf.tile(tf.expand_dims(tf.gather_nd(tf.transpose(res), [0]), axis=1), multiples=[1, 2])
        row_indices = tf.range(tf.shape(y_true)[0])
        indices = tf.gather_nd(tf.transpose(y_true), [1])
        full_indices = tf.stack([tf.cast(row_indices, dtype=tf.int32), tf.cast(indices, dtype=tf.int32)], axis=1)
        sparse = tf.SparseTensor(indices=tf.cast(full_indices, dtype=tf.int64), values=res, dense_shape=[4, 2])
        res = tf.sparse.to_dense(sparse)

        # return tf.keras.losses.huber(q_true, values)
        res = tf.reduce_mean(res, axis=0)

        # tf.print(res)

        return res


class Network:
    def __init__(self, env, args, training=True):
        self._model = tf.keras.models.Sequential()
        self._model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation="relu",  input_dim=env.observation_space.shape[0], name="d1"))
        # self._model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation="relu"))
        # self._model.add(tf.keras.layers.Dense(args.hidden_layer_size*2, activation="relu"))
        self._model.add(tf.keras.layers.Dense(env.action_space.n, activation="relu", name="d4"))  # activation="softmax"
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        if training:
            self._model.compile(optimizer=opt, loss=MeanValueSquaredError())  # MeanValueSquaredError()
        # it is stored as `self._model` and has been `.compile()`-d.

    # Training method: Generally you have two possibilities: code below implements first option, you can change it ...

    # q values of all actions for given state (aka. output of net)
    # ...  all but one are the same as before ...

    # one new q_value for a given state ... +include index of action to which the new q_value belongs ...
    @tf.function
    def train(self, states, q_values):
        self._model.optimizer.minimize(
            lambda: self._model.loss(q_values, self._model(states, training=True)),  # q_old, q_new
            var_list=self._model.trainable_variables
        )

    # Predict method, again with manual @tf.function for efficiency.
    @tf.function
    def predict(self, states):
        return self._model(states)

    # If you want to use target network, the following method copies weights from a given Network to the current one.
    def copy_weights_from(self, other):
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(other_var)


def main(env, args):
    episode_decay = np.power(args.epsilon_final/args.epsilon, 1/args.epsilon_final_at)

    generator = np.random.RandomState(args.seed)

    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)
    fixed_network = Network(env, args)
    fixed_network.copy_weights_from(network)

    # Replay memory;
    # maxl en parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    memory = collections.deque()

    Transition = collections.namedtuple("Transition",
                                        ["state", "action", "reward", "done", "next_state"]
                                        )

    epsilon = args.epsilon
    if args.recodex:
        training = False
        with zipfile.ZipFile("my_model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    else:
        training = True

    episode_step = 0
    returns = []
    best_res = 400
    while training:

        if len(returns) > 100 and np.average(returns[-100:]) > best_res:
            best_res = np.average(returns[-100:])
            print("BEST RESULT: {}".format(best_res))
            network._model.save("my_model", include_optimizer=False)

        # Perform episode
        state, done = env.reset(), False
        if episode_step and episode_step % args.target_update_freq == 0:
            fixed_network.copy_weights_from(network)
            print("weights updated")
        if episode_step:
            epsilon = max(epsilon * args.epsilon_decay, 0.0001)
        episode_step += 1

        epside_return = 0
        while not done:
            # epsilon-greedy policy
            if generator.uniform() < epsilon:
                action = generator.randint(env.action_space.n)
            else:
                q_values = network.predict(np.array([state], np.float32))[0]
                action = np.argmax(q_values)

            # Perform step
            next_state, reward, done, _ = env.step(action)
            epside_return += reward
            # Append state, action, reward, done and next_state to replay_buffer
            memory.append(Transition(state, action, reward, done, next_state))

            # if replay buffer large enough ...
            if len(memory) >= 1000:
                idxs = generator.choice(np.arange(len(memory)), args.batch_size)
                states = [memory[i].state for i in idxs]
                targets = [[memory[i].reward + args.gamma * np.max(fixed_network.predict(np.asarray([memory[i].next_state]))[0]),  memory[i].action]
                           if not memory[i].done else
                           [memory[i].reward, memory[i].action]
                           for i in idxs]

                states = np.asarray(states)  # (32, 4)
                targets = np.asarray(targets)  # (32, 2)

                # Choose `states` and suitable targets
                network.train(states, targets)
                # exit(1)
                memory.popleft()

            state = next_state

        returns.append(epside_return)
        # if args.epsilon_final_at:
        #     epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

    print("LOADING")
    network = Network(env, args, training=True)
    network._model = tf.keras.models.load_model("my_model")
    print("LOADED")
    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            action = np.argmax(network.predict(np.asarray([state]))[0])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPole-v1"), args.seed)
    main(env, args)
