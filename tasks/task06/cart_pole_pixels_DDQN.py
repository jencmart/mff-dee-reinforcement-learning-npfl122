#!/usr/bin/env python3
import argparse
import collections
import os
import zipfile
import matplotlib.pyplot as plt
import gym
import numpy as np
import tensorflow as tf
import wrappers
import cart_pole_pixels_environment

# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")  # Report only TF errors by default

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Frame skip.")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Frame skip.")
parser.add_argument("--gamma", default=0.99, type=float, help="Frame skip.")

parser.add_argument("--epsilon", default=0.1, type=float, help="Frame skip.")
parser.add_argument("--epsilon_decay", default=0.9993, type=float, help="Frame skip.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Frame skip.")

parser.add_argument("--target_update_freq", default=200, type=int, help="Frame skip.")
parser.add_argument("--buffer_init_len", default=300, type=int, help="Frame skip.")
parser.add_argument("--maxlen", default=3000, type=int, help="Frame skip.")
parser.add_argument("--actions", default=2, type=int, help="Frame skip.")
parser.add_argument("--image_size", default=2, type=int, help="Frame skip.")

parser.add_argument("--l2", default=0.001, type=float, help="Frame skip.")
parser.add_argument("--omega", default=0.6, type=float, help="Frame skip.")
parser.add_argument("--beta", default=0.5, type=float, help="Frame skip.")

parser.add_argument("--kernel_size", default=16, type=int, help="Frame skip.")
parser.add_argument("--filters", default=16, type=int, help="Frame skip.")
parser.add_argument("--gru_units", default=16, type=int, help="Frame skip.")
parser.add_argument("--state_value_prediction_units", default=216, type=int, help="Frame skip.")
parser.add_argument("--advantage_units", default=64, type=int, help="Frame skip.")


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
                                 dense_shape=[full_indices.shape[0], args.actions])
        res = tf.sparse.to_dense(sparse)

        res = tf.reduce_mean(res, axis=0)
        return res


def plot_data(means_returns):
    x_data = np.linspace(0, len(means_returns), len(means_returns))
    plt.plot(x_data, means_returns, 'r-')
    plt.draw()
    plt.pause(0.00001)
    plt.clf()


class Network:
    def __init__(self, env, args):
        output_size = env.action_space.n
        input_shape = list(env.observation_space.shape)
        input_shape[-1] = 1
        all_inputs = [
            tf.keras.Input(shape=input_shape),
            tf.keras.Input(shape=input_shape),
            tf.keras.Input(shape=input_shape)
        ]
        layers = [
            tf.keras.layers.Conv2D(args.filters, args.kernel_size, activation=tf.nn.relu, strides=4),  # originally stride = 3
            tf.keras.layers.MaxPool2D(4, 4),
            # tf.keras.layers.Conv2D(args.filters * 2, args.kernel_size // 2, activation=tf.nn.relu),
            # tf.keras.layers.MaxPool2D(2, 2),
            # tf.keras.layers.Conv2D(args.filters * 4, args.kernel_size // 4, activation=tf.nn.relu),
            # tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation=tf.nn.relu)
        ]
        cnn_outputs = []
        for input_layer in all_inputs:
            for conv_layer in layers:
                input_layer = conv_layer(input_layer)
            cnn_outputs.append(input_layer)

        net = tf.stack(cnn_outputs, 1)
        # net = tf.reduce_sum(net, 1)
        # net = tf.keras.layers.Dense(units=args.gru_units, return_sequences=False)(net)
        net = tf.keras.layers.GRU(units=args.gru_units, return_sequences=False)(net)

        # state value prediction
        state_prediction = tf.keras.layers.Dense(args.state_value_prediction_units, activation=tf.nn.relu)(net)
        state_prediction = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(state_prediction)
        replicate_state_value = lambda x: tf.transpose(
            tf.reshape(tf.tile(tf.reshape(x, [-1]), [output_size]), [output_size, tf.shape(x)[0]]))
        tile_state_prediction_layer = tf.keras.layers.Lambda(replicate_state_value, output_size)(state_prediction)
        #
        # # action advantage prediction
        action_adv = tf.keras.layers.Dense(64, activation=tf.nn.relu)(net)
        action_adv = tf.keras.layers.Dense(output_size, activation=tf.nn.relu)(action_adv)
        substract_mean = lambda x: x - tf.transpose(
            tf.reshape(tf.tile(tf.reduce_mean(x, axis=1), [output_size]), [output_size, tf.shape(x)[0]]))
        substract_average_layer = tf.keras.layers.Lambda(substract_mean, output_shape=output_size)(action_adv)
        # #
        output = tf.keras.layers.Add()([substract_average_layer, tile_state_prediction_layer])
        output = tf.keras.layers.Dense(output_size)(output)  # activation=tf.nn.relu,

        self._model = tf.keras.Model(all_inputs, output)
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=MeanValueSquaredError()
        )

    def prepare_states(self, states):
        states1 = states[:, :, :, 0]
        states1 = tf.expand_dims(states1, axis=3)
        states2 = states[:, :, :, 1]
        states2 = tf.expand_dims(states2, axis=3)
        states3 = states[:, :, :, 2]
        states3 = tf.expand_dims(states3, axis=3)
        return [states1, states2, states3]

    @tf.function
    def train(self, states, q_values):
        # can do rescaling ... too large image
        # states = tf.image.resize(states, [args.image_size, args.image_size]).numpy()
        self._model.optimizer.minimize(
            lambda: self._model.loss(q_values, self._model(self.prepare_states(states), training=True)),  # q_old, q_new
            var_list=self._model.trainable_variables
        )

    # Predict method, again with manual @tf.function for efficiency.
    @tf.function
    def predict(self, states):
        return self._model(self.prepare_states(states))

    # If you want to use target network, the following method copies weights from a given Network to the current one.
    def copy_weights_from(self, other):
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(other_var)


# return tuple (action_value, action_index)
def choose_action(q_value, generator, epsilon, args):
    action = np.argmax(q_value)
    if epsilon is not None and generator.uniform() < epsilon:
        action = np.random.randint(0, args.actions)
    return action, action


# returns targets and states
def priority_sampling(weights, memory, args, nn, fixed_nn):
    transformed_weights = np.array([weights]) ** args.omega
    batch_indices = np.random.choice(len(memory), args.batch_size, replace=False, p=np.array(transformed_weights)/sum(transformed_weights))
    updated_weights, targets, states, dynamic_learning_rates = [], [], [], []
    rhos = np.array((len(memory) * weights) ** (-1 * args.beta))
    rhos /= np.max(rhos)

    for i in batch_indices:
        s, a, r, d, ns = memory[i]
        importance_sampling_weight = rhos[i]
        fixed_predictions = fixed_nn.predict(np.asarray([ns]))[0]
        index = np.argmax(nn.predict(np.asarray([ns]))[0])
        # compute error and store it in the weights buffer
        error = r + args.gamma * fixed_predictions[index] - nn.predict(np.asarray([s]))[0][a]
        updated_weights.append([i, np.abs(error)])
        dynamic_learning_rates.append(importance_sampling_weight)
        targets.append([r, a])
        states.append(s)
        if not d:
            targets[-1][0] += args.gamma * fixed_predictions[index]

    for index, new_weight in updated_weights:
        weights[index] = new_weight
    return np.asarray(states), np.asarray(targets), np.array(dynamic_learning_rates)


def create_states_and_targets(memory, network, fixed_network, batch_size, gamma):
    indices = np.random.choice(np.arange(len(memory)), batch_size, replace=False)
    targets, states = [], []
    for i in indices:
        s, a, r, d, ns = memory[i]
        if not d:
            fixed_predictions = fixed_network.predict(np.asarray([ns]))[0]
            index = np.argmax(network.predict(np.asarray([ns]))[0])
            tmp = [r + gamma * fixed_predictions[index], a]
        else:
            tmp = [r, a]
        targets.append(tmp)
        states.append(s)
    return np.asarray(states), np.array(targets)


def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    generator = np.random.RandomState(args.seed)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])
    memory = collections.deque(maxlen=args.maxlen)
    weights = collections.deque(maxlen=args.maxlen)

    if args.recodex:
        with zipfile.ZipFile("my_model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

        network = Network(env, args)
        network._model = tf.keras.models.load_model("my_model")
        while True:
            state, done = env.reset(start_evaluation=True), False
            while not done:
                q_values = network.predict(np.array([state], np.float32))[0]
                action, idx_action = choose_action(q_values, generator, -1, args)
                next_state, reward, done, _ = env.step(action)
                memory.append(Transition(state, idx_action, reward, done, next_state))
                state = next_state
    else:
        network = Network(env, args)
        fixed_network = Network(env, args)
        fixed_network.copy_weights_from(network)

        episode_step, best_so_far, epsilon = 0, 100, args.epsilon
        returns, means_returns = [], []
        while True:
            # Perform episode
            state, done = env.reset(), False
            current_return = 0
            while not done:
                if args.render_each and env.episode and env.episode % args.render_each == 0: env.render()
                q_values = network.predict(np.array([state], np.float32))[0]
                action, idx_action = choose_action(q_values, generator, epsilon, args)

                # Perform step
                next_state, reward, done, _ = env.step(action)
                current_return += reward
                memory.append(Transition(state, idx_action, reward, done, next_state))
                state = next_state

                # Train
                if len(memory) >= args.buffer_init_len:
                    states, targets = create_states_and_targets(memory, network, fixed_network, args.batch_size, args.gamma)
                    network.train(states, targets)

            returns.append(current_return)
            if episode_step % args.target_update_freq == 0:
                fixed_network.copy_weights_from(network)
            epsilon = max(epsilon * args.epsilon_decay, args.epsilon_final)
            episode_step += 1

            # Save weights ...
            if len(returns) > 100:
                if np.average(returns[-100:]) > best_so_far + 10:
                    best_so_far = np.average(returns[-100:])
                    network._model.save("my_model", include_optimizer=False)
                    print("Best so far: {}".format(best_so_far))


if __name__ == "__main__":
    args = parser.parse_args([] if "_file_" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPolePixels-v0"), args.seed)
    main(env, args)