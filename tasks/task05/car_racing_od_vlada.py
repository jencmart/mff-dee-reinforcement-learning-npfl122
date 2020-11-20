#!/usr/bin/env python3
import argparse
import collections
import os
import gym
import numpy as np
import tensorflow as tf
import car_racing_environment
import wrappers

# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Report only TF errors by default

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=10, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--frame_skip", default=5, type=int, help="Frame skip.")
parser.add_argument("--frame_stack", default=1, type=int, help="Frame skip.")
parser.add_argument("--cnt_steer", default=6, type=int, help="Frame skip.")
parser.add_argument("--cnt_gas", default=5, type=int, help="Frame skip.")
parser.add_argument("--cnt_brake", default=5, type=int, help="Frame skip.")
parser.add_argument("--batch_size", default=8, type=int, help="Frame skip.")  # 64 for ref todo 16
parser.add_argument("--learning_rate", default=0.001, type=float, help="Frame skip.")
parser.add_argument("--gamma", default=0.999, type=float, help="Frame skip.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Frame skip.")
parser.add_argument("--epsilon_decay", default=0.995, type=float, help="Frame skip.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Frame skip.")
parser.add_argument("--target_update_freq", default=15, type=int, help="Frame skip.")
parser.add_argument("--buffer_init_len", default=200, type=int, help="Frame skip.")
parser.add_argument("--l2", default=0.001, type=float, help="Frame skip.")
parser.add_argument("--omega", default=0.6, type=float, help="Frame skip.")
parser.add_argument("--beta", default=0.5, type=float, help="Frame skip.")


class MeanValueSquaredError(tf.keras.losses.Loss):
    @tf.function
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
        sparse = tf.SparseTensor(indices=tf.cast(full_indices, dtype=tf.int64), values=res,
                                 dense_shape=[full_indices.shape[0], 20])
        res = tf.sparse.to_dense(sparse)

        # return tf.keras.losses.huber(q_true, values)
        res = tf.reduce_mean(res, axis=0)
        # tf.print(res)
        return res


class Network:
    def __init__(self, env, args):
        input_size = list(env.observation_space.shape)
        input_size[-1] = args.frame_stack
        reg = None
        output_size = args.cnt_brake + args.cnt_gas + args.cnt_steer

        inp = tf.keras.layers.Input(input_size)
        net = tf.keras.layers.Conv2D(8, (7, 7), activation=tf.nn.relu, strides=3, )(inp)
        net = tf.keras.layers.MaxPool2D(2, 2)(net)
        net = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, )(net)

        net = tf.keras.layers.MaxPool2D(2, 2)(net)
        net = tf.keras.layers.Flatten()(net)
        # net = tf.keras.layers.Dense(256, activation=tf.nn.relu, )(net)

        # state value prediction
        state_prediction = tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=reg)(net)
        state_prediction = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer=reg)(state_prediction)
        replicate_state_value = lambda x: tf.transpose(tf.reshape(tf.tile(tf.reshape(x, [-1]), [output_size]), [output_size, tf.shape(x)[0]]))
        tile_state_prediction_layer = tf.keras.layers.Lambda(replicate_state_value, output_size)(state_prediction)
        #
        # # action advantage prediction
        action_adv = tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=reg)(net)
        action_adv = tf.keras.layers.Dense(output_size, activation=tf.nn.relu, kernel_regularizer=reg)(action_adv)
        substract_mean = lambda x: x - tf.transpose(tf.reshape(tf.tile(tf.reduce_mean(x, axis=1), [output_size]), [output_size, tf.shape(x)[0]]))
        substract_average_layer = tf.keras.layers.Lambda(substract_mean, output_shape=output_size)(action_adv)
        #
        output = tf.keras.layers.Add()([substract_average_layer,  tile_state_prediction_layer])
        output = tf.keras.layers.Dense(output_size, activation=tf.nn.relu, kernel_regularizer=reg)(output)

        self._model = tf.keras.Model(inp, output)
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=MeanValueSquaredError()
        )

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


# return tuple (action_value, action_index)
def choose_action(q_value, generator, epsilon, args):
    action_cnt = len(q_value)
    action = np.argmax(q_value)
    if generator.uniform() < epsilon:
        action = generator.randint(action_cnt)

    acc = args.cnt_steer
    if 0 <= action < acc:
        return [np.linspace(-0.9, 0.9, args.cnt_steer)[action], 0, 0], action

    acc += args.cnt_gas
    if args.cnt_steer <= action < acc:
        return [0, np.linspace(0.1, 0.9, args.cnt_gas)[action - args.cnt_steer], 0], action

    acc += args.cnt_brake
    if args.cnt_gas <= action < acc:
        return [0, 0, np.linspace(0.1, 0.9, args.cnt_brake)[action - args.cnt_steer - args.cnt_gas]], action

    raise NotImplemented('ah shit, here we go again')


def rgb2gray(rgb):
    return np.expand_dims(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]), axis=2)


def create_stacked(state, memory, args):
    if args.frame_stack == 1:
        return state
    if len(memory) == 0:
        stacked_state = state
        for i in range(args.frame_stack - 1):
            stacked_state = np.append(stacked_state, state, axis=2)
    else:
        stacked_state = memory[len(memory) - 1 - args.frame_stack + 1 + 1].state if len(memory) - 1 - args.frame_stack + 1 + 1 > 0 else memory[0].state
        for i in range(len(memory) - 1 - args.frame_stack + 1 + 1 + 1, max(len(memory), args.frame_stack + 1)):
            if stacked_state.shape[2] == args.frame_stack:
                break
            if i < 0:
                stacked_state = np.append(stacked_state, memory[0].state, axis=2)
            if i > len(memory):
                stacked_state = np.append(stacked_state, memory[-1].state, axis=2)
    return stacked_state


# returns targets and states
def priority_sampling(weights, memory, args, network, fixed_network, generator):
    states, targets = [], []

    priorities = np.array(weights) ** args.omega
    priorities_sum = np.sum(priorities)
    priorities_norm = priorities / priorities_sum

    # rhos are importance sampling weights
    rhos = np.array([(p * len(weights)) ** (-args.beta) for p in priorities_norm])
    rhos = rhos / np.max(rhos)
    rhos_norm = rhos / np.sum(rhos)

    for _ in range(args.batch_size):
        # fix nans
        if np.isnan(priorities_norm).any():
            # print('Ah shit, here we go again : nans priorities_norm')
            priorities_norm[np.argwhere(np.isnan(priorities_norm))] = 0.0

        if np.sum(priorities_norm) != 1:
            # print('Ah shit, here we go again : !=1 priorities_norm')
            priorities_norm += (1 - np.sum(priorities_norm)) / len(priorities_norm)

        i = generator.choice(len(memory), p=priorities_norm)
        s, a, r, d, ns = memory[i]

        # fix nans
        if np.isnan(rhos_norm).any():
            # print('Ah shit, here we go again : rhos_norm')
            rhos_norm[np.argwhere(np.isnan(rhos_norm))] = 0.0
        if np.sum(rhos_norm) != 1:
            # print('Ah shit, here we go again : !=1 rhos_norm')
            rhos_norm += (1 - np.sum(rhos_norm)) / len(rhos_norm)

        rho = generator.choice(len(weights), p=rhos_norm)
        fixed_predictions = fixed_network.predict(np.asarray([ns]))[0]
        index = np.argmax(network.predict(np.asarray([ns]))[0])
        # compute error and store it in the weights buffer
        error = r + args.gamma * fixed_predictions[index] - network.predict(np.asarray([s]))[0][a]
        weights[i] = np.abs(error)

        # update priorities
        # new priority in temp. variable
        t = weights[i] ** args.omega
        # update sum ... subtract old priority and add new
        priorities_sum = priorities_sum - priorities[i] + t
        # update new priority
        priorities[i] = t
        # update normalized priorities
        priorities_norm = priorities / priorities_sum

        # update rhos
        rhos[i] = (priorities_norm[i] * len(weights)) ** -args.beta
        rhos = rhos / np.max(rhos)
        rhos_norm = rhos / np.sum(rhos)

        states.append(s)
        targets.append([r * rho, a])
        if not d:
            targets[-1][0] += rho * args.gamma * fixed_predictions[index]
    return np.asarray(states), np.asarray(targets)


def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    generator = np.random.RandomState(args.seed)

    if args.recodex:
        # TODO: Perform evaluation of a trained model.

        while True:
            state, done = env.reset(start_evaluation=True), False
            while not done:
                # TODO: Choose an action
                action = None
                state, reward, done, _ = env.step(action)

    else:
        # Replay memory
        memory = collections.deque()
        weights = collections.deque()
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])
        # Construct the network
        network = Network(env, args)
        fixed_network = Network(env, args)
        fixed_network.copy_weights_from(network)

        epsilon = args.epsilon
        episode_step = 0
        returns = []
        while True:
            # Perform episode
            state, done = env.reset(), False
            state = rgb2gray(state)
            current_return = 0

            while not done:
                if args.render_each and env.episode and env.episode % args.render_each == 0: env.render()

                state = create_stacked(state, memory, args)
                q_values = network.predict(np.array([state], np.float32))[0]
                action, idx_action = choose_action(q_values, generator, epsilon, args)

                # Perform step
                next_state, reward, done, _ = env.step(action)
                next_state = create_stacked(rgb2gray(next_state), memory, args)
                current_return += reward
                memory.append(Transition(state, idx_action, reward, done, next_state))
                weights.append(np.max(weights) if len(weights) > 0 else 1)
                if len(memory) >= args.buffer_init_len:
                    indices = generator.choice(np.arange(len(memory)), args.batch_size, replace=False)
                    # states, targets = priority_sampling(weights, memory, args, network, fixed_network, generator)
                    targets, states = [], []
                    for i in indices:
                        s, a, r, d, ns = memory[i]
                        if not d:
                            fixed_predictions = fixed_network.predict(np.asarray([ns]))[0]
                            index = np.argmax(network.predict(np.asarray([ns]))[0])
                            tmp = [r + args.gamma * fixed_predictions[index], a]
                        else:
                            tmp = [r, a]
                        targets.append(tmp)
                        states.append(s)
                    # print(targets)
                    network.train(np.array(states), np.array(targets))

                state = next_state
            returns.append(current_return)
            if episode_step % args.target_update_freq == 0:
                fixed_network.copy_weights_from(network)
            epsilon = max(epsilon * args.epsilon_decay, args.epsilon_final)
            episode_step += 1

            if len(returns) > 1000 and np.average(returns[-500:]) >= 430:
                print(f'Training finished with {np.average(returns[-500:])}')
                break

        network._model.save("my_model", include_optimizer=False)
        print("LOADING")


if __name__ == "__main__":
    args = parser.parse_args([] if "file" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CarRacingSoftFS{}-v0".format(args.frame_skip)), args.seed,
                                     evaluate_for=15, report_each=1)

    main(env, args)