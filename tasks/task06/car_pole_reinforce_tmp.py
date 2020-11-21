#!/usr/bin/env python3
import argparse
import os
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default
import gym
import numpy as np
import tensorflow as tf
import cart_pole_pixels_environment
import wrappers
import utils

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--use_baseline", default=True, type=bool, help="Learning rate.")

parser.add_argument("--batch_size", default=12, type=int, help="Number of episodes to train on.")
parser.add_argument("--gamma", default=0.983, type=float, help="gamma for discount.")
parser.add_argument("--learning_rate", default=0.00898, type=float, help="Learning rate.")
parser.add_argument("--hidden_layer", default=64, type=int, help="Size of hidden layer.")  # originally 238
parser.add_argument("--hidden_layer_baseline", default=32, type=int, help="Size of hidden layer.")  # originally 220
parser.add_argument("--learning_rate_baseline", default=0.007, type=float, help="Learning rate.")
parser.add_argument("--image_size", default=80, type=int, help="Learning rate.")


def get_simple_cnn_GRU(input_shape, output_classes):
    input_shape = [input_shape[0], input_shape[1], 1]
    in1 = tf.keras.Input(shape=input_shape)
    in2 = tf.keras.Input(shape=input_shape)
    in3 = tf.keras.Input(shape=input_shape)
    all_inputs = [in1, in2, in3]
    cnn_outputs = []
    layers = [tf.keras.layers.Conv2D(4, kernel_size=(4, 4), strides=2, activation='relu', padding='same',
                                     input_shape=input_shape),
              tf.keras.layers.Conv2D(8, (4, 4), strides=2, padding='same', activation='relu'),
              tf.keras.layers.Conv2D(16, (4, 4), strides=2, padding='same', activation='relu'),
              tf.keras.layers.Flatten()]

    for input in all_inputs:
        for layer in layers:
            input = layer(input)
        cnn_outputs.append(input)

    x = tf.stack(cnn_outputs, 1)

    gru_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=16, return_sequences=True), merge_mode='sum')(x)
    gru_out = tf.keras.layers.Flatten()(gru_out)
    output = tf.keras.layers.Dense(output_classes, activation='softmax')(gru_out)
    model = tf.keras.Model(all_inputs, [output])
    return model


class BaselineNetwork:
    def __init__(self, shape, classes, args):
        model = get_simple_cnn_GRU(shape, classes)
        self.model = model
        self.model.compile(optimizer=tf.optimizers.Adam(lr=args.learning_rate_baseline), loss='mse')

    def train(self, states, returns):
        self.model.train_on_batch(x=states, y=returns)

    def predict(self, states):
        # states = np.array(states, np.float32)
        results = self.model.predict_on_batch(x=states)
        if isinstance(results, np.ndarray):
            pass
        else:
            results = results.numpy()
        results = results[:, 0]
        return results


class Network:
    def __init__(self, env, args):
        states_shape = [args.image_size, args.image_size, 3]  # env.state_shape  [80,80,3]
        self.resize_to = args.image_size
        self.use_baseline = args.use_baseline
        classes = env.action_space.n
        model = get_simple_cnn_GRU(states_shape, classes)
        self.model = model
        self.model.compile(optimizer=tf.optimizers.Adam(lr=args.learning_rate), loss='categorical_crossentropy')  # mse
        if self.use_baseline:
            self.baseline_network = BaselineNetwork(shape=states_shape, classes=1, args=args)

    def deal_with_states(self, states):
        # [batch, 80, 80, 3] -> [batch,x,x,3]
        states = tf.image.resize(states, [self.resize_to, self.resize_to]).numpy()

        states1 = states[:, :, :, 0]
        states1 = np.expand_dims(states1, axis=3)
        states2 = states[:, :, :, 1]
        states2 = np.expand_dims(states2, axis=3)
        states3 = states[:, :, :, 2]
        states3 = np.expand_dims(states3, axis=3)
        return [states1, states2, states3]

    def train(self, states, actions, returns):
        # mnist = 28x28
        # cifar = 32x32
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns,
                                                                                                       np.float32)
        # Train the model using the states, actions and observed returns by calling `train_on_batch`.
        # States  [ batch , 4 ]
        # Actions [ batch , 2 ]
        # returns [ batch , 1 ]

        states = self.deal_with_states(states)

        onehot_actions = np.zeros((actions.size, 2), dtype=np.int32)
        onehot_actions[np.arange(actions.size), actions] = 1

        # todo: first train the baseline
        if self.use_baseline:
            self.baseline_network.train(states, returns)

        # todo: predict baseline using baseline network
        if self.use_baseline:
            baseline = self.baseline_network.predict(states)
            returns -= baseline

        self.model.train_on_batch(x=states, y=onehot_actions, sample_weight=returns)

    def predict(self, states):
        states = np.array(states, np.float32)
        states = self.deal_with_states(states)

        # Predict distribution over actions for the given input states
        # using the `predict_on_batch` method and calling `.numpy()` on the result to return a NumPy array.
        results = self.model.predict_on_batch(x=states)
        if isinstance(results, np.ndarray):
            pass
        else:
            results = results.numpy()
        return results


def calculate_returns(rewards, gamma=0.99, subs_mean=False):
    returns = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
    returns = np.cumsum(returns[::-1])[::-1]
    if subs_mean:
        returns -= returns.mean()
    return returns


def test(arg, environment, net_name, baseline_net_name):
    network = Network(environment, arg)
    network.model = utils.load_model(net_name)
    network.baseline_network.model = utils.load_model(baseline_net_name)
    while True:
        state, done = environment.reset(start_evaluation=True), False
        while not done:
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = environment.step(action)


def main(env, args):
    network_name = "main_network"
    baseline_network_name = "baseline_network"
    init_save_threshold = 300

    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if args.recodex:
        test(args, env, network_name, baseline_network_name)

    else:
        # Construct the network
        network = Network(env, args)
        network_saver = utils.Saver(init_save_threshold, network.model, network_name)
        baseline_network_saver = utils.Saver(init_save_threshold, network.baseline_network.model, baseline_network_name)

        possible_actions = list(range(env.action_space.n))
        episode_returns, avg_returns = [], []
        # Training
        while True:
            batch_states, batch_actions, batch_returns = [], [], []
            for _ in range(args.batch_size):
                # Perform episode
                states, actions, rewards = [], [], []
                state, done = env.reset(), False
                er = 0
                while not done:
                    if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                        env.render()

                    probabilities = network.predict([state])[0]
                    action = np.random.choice(a=possible_actions, p=probabilities)
                    next_state, reward, done, _ = env.step(action)
                    er += reward
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    state = next_state
                episode_returns.append(er)
                returns = calculate_returns(rewards, gamma=args.gamma, subs_mean=False)
                batch_states += states
                batch_actions += actions
                batch_returns += returns.tolist()
            # avg_returns.append(np.mean(np.asarray(episode_returns[-100:])))
            avg_returns.append(sum(episode_returns[-100:])/100)
            utils.save_and_plot(avg_returns, network_saver, baseline_network_saver)
            # if avg_returns[-1] >= 500:
            #     break

            network.train(batch_states, batch_actions, batch_returns)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPolePixels-v0"), args.seed)

    main(env, args)
