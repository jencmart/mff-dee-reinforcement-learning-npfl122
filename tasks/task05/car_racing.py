#!/usr/bin/env python3
import argparse
import collections
import os
import time
import zipfile
from os.path import basename

import gym
import numpy as np
import tensorflow as tf
import car_racing_environment
import wrappers
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Report only TF errors by default

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=4, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--frame_stack", default=3, type=int, help="Frame skip.")
parser.add_argument("--batch_size", default=8, type=int, help="Frame skip.")  # 64 for ref todo 16
parser.add_argument("--learning_rate", default=0.001, type=float, help="Frame skip.")
parser.add_argument("--gamma", default=0.95, type=float, help="Frame skip.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Frame skip.")
parser.add_argument("--epsilon_decay", default=0.992, type=float, help="Frame skip.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Frame skip.")
parser.add_argument("--target_update_freq", default=15, type=int, help="Frame skip.")


parser.add_argument("--maxlen", default=5000, type=int, help="Frame skip.")
parser.add_argument("--save_if_return", default=-10, type=float, help="Frame skip.")
parser.add_argument("--model_name", default="my_model", type=str, help="Frame skip.")
parser.add_argument("--renderable", default=True, type=bool, help="Frame skip.")
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])


class ActionSpace:
    # 12 akci
    action_space = [
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Steer + Full Gas + Brake
        (-1, 1, 0), (0, 1, 0), (1, 1, 0),        # Steer + Full Gas

        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Steer + Brake
        (-1, 0, 0), (0, 0, 0), (1, 0, 0)         # Steer
    ]

    def __init__(self):
        pass


class Network:

    def __init__(self, args, output_size):
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        self.memory = collections.deque(maxlen=args.maxlen)
        self._model = self.build_network(args, output_size)
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate, epsilon=1e-7),
            loss=tf.keras.losses.MeanSquaredError()  # MeanValueSquaredError()
        )

        self._fixed_network = self.build_network(args, output_size)
        self._copy()

    def build_network(self, args, output_size):
        input_size = list(env.observation_space.shape)
        input_size[-1] = args.frame_stack
        reg = None

        inp = tf.keras.layers.Input(input_size)
        net = tf.keras.layers.Conv2D(6, (7, 7), activation=tf.nn.relu, strides=3, )(inp)
        net = tf.keras.layers.MaxPool2D(2, 2)(net)
        net = tf.keras.layers.Conv2D(12, (4, 4), activation=tf.nn.relu, )(net)
        net = tf.keras.layers.MaxPool2D(2, 2)(net)
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(216, activation=tf.nn.relu, )(net)
        output = tf.keras.layers.Dense(output_size, activation=tf.nn.softmax)(net)
        return tf.keras.Model(inp, output)

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
    def _copy(self):
        for fixed_var, other_var in zip(self._fixed_network.variables, self._model.variables):
            fixed_var.assign(other_var)
    def copy_weights_to_fixed(self):
        self._copy()


    def add_to_memory(self, stacked_state, idx_action, reward, done, stacked_next_state):
        self.memory.append(
            Transition(stacked_state, idx_action, reward, done, stacked_next_state))

    def train_on_batch(self):
        # 1. Prepare targets ...
        indices = np.random.choice(np.arange(len(self.memory)), self.batch_size, replace=False)
        targets, states = [], []
        for i in indices:
            s, a_idx, reward, done, ns = self.memory[i]

            prediction = self.predict(np.expand_dims(s, axis=0))[0]
            prediction = np.copy(prediction)
            fixed_prediction = self._fixed_network.predict(np.expand_dims(ns, axis=0))[0]
            # q_max = np.max(fixed_prediction)
            q_max = fixed_prediction[np.argmax(prediction)]
            if done:
                prediction[a_idx] = reward
            else:
                prediction[a_idx] = reward + self.gamma * q_max  # q(s,a) = r + g * max_a [  q(s_next, a) ]
            targets.append(prediction)
            states.append(s)

        # 2. Train
        states = np.array(states)
        targets = np.array(targets)
        # print(states.shape)
        # print(targets.shape)
        self.was_trained = True
        self.train(states, targets)


def choose_action(q_value, epsilon):
    if epsilon is not None and np.random.uniform() < epsilon:
        action_idx = np.random.randint(len(ActionSpace.action_space))
    else:
        action_idx = int(np.argmax(q_value))

    return action_idx


def rgb2gray(rgb):
    return np.expand_dims(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]), axis=2)


def create_stacked(state, memory, frames, prev_im=None):
    stacked_state = state

    rng = frames-1
    if prev_im is not None and rng > 0:  # do not even include previous if frame_stack=1
        if rng > 0:
            unstacked = np.expand_dims(prev_im[:,:,0], axis=2)
            stacked_state = np.append(stacked_state, unstacked, axis=2)
        rng -= 1  # previous image was, used .., one less to choose from memory

    for i in range(rng):
        idx = len(memory)-1-i
        if idx < 0:
            if len(memory) == 0:
                stacked_state = np.append(stacked_state, state, axis=2)
            else:
                unstacked = np.expand_dims(memory[0].state[:,:,0], axis=2)
                stacked_state = np.append(stacked_state, unstacked, axis=2)
        else:
            unstacked = np.expand_dims(memory[idx].state[:, :, 0], axis=2)

            stacked_state = np.append(stacked_state, unstacked, axis=2)
    return stacked_state


def get_action(stacked_state, network, epsilon):
    q_values = network.predict(np.array([stacked_state], np.float32))[0]
    return choose_action(q_values, epsilon)


def log(name, epsiode, avg_returns):
    avg_return = avg_returns[-1]
    with open(name, "a") as f:
        f.write("E: {}, avg_ret: {}\n".format(epsiode, avg_return))


def perform_testing():
    with zipfile.ZipFile("{}.zip".format(args.model_name), 'r') as zip_ref:
        zip_ref.extractall(".")
    network = Network(args, len(ActionSpace.action_space))
    network._model = tf.keras.models.load_model(args.model_name)
    while True:
        state, done = rgb2gray(env.reset(start_evaluation=True)), False
        while not done:
            if args.renderable and args.render_each and env.episode and env.episode % args.render_each == 0: env.render()
            state, reward, done, _ = env.step(ActionSpace.action_space[get_action(create_stacked(state, network.memory,  args.frame_stack), network, None)])
            state = rgb2gray(state)


def plot_data(means_returns, dotsx, dotsy):
    x_data = np.linspace(0, len(means_returns), len(means_returns))
    plt.plot(x_data, means_returns, 'r-')
    plt.plot(dotsx, dotsy, 'bo')
    plt.draw()
    plt.pause(0.00001)
    plt.clf()



def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if args.recodex:
        perform_testing()
    else:
        # INITIALIZE TRAINING
        log_name = args.model_name + str(int(time.time())) + ".log"
        # Construct the network
        network = Network(args, len(ActionSpace.action_space))
        network.copy_weights_to_fixed()
        # For updating epsilon and copying weights between networks
        epsilon, episodes, best_episode = args.epsilon, 0, args.save_if_return
        # Returns ...
        dotsx, dotsy, means_returns, returns = [], [], [], []

        while True:
            # RUN EPISODE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            state, done, episode_return, frame, neg_counter, trained_frames = rgb2gray(env.reset()), False, 0, 1, 0, 0
            while True:
                # print(frame)
                if args.renderable and args.render_each and env.episode and env.episode % args.render_each == 0: env.render()
                # Predict action ...
                stacked_state = create_stacked(state, network.memory,  args.frame_stack)
                action_idx = get_action(stacked_state, network, None)

                # Perform step and memorize...
                next_state, reward, done, _ = env.step(ActionSpace.action_space[action_idx])
                next_state = rgb2gray(next_state)
                network.add_to_memory(stacked_state, action_idx, reward, done, create_stacked(next_state, network.memory,  args.frame_stack, stacked_state))
                episode_return += reward
                state = next_state

                neg_counter = neg_counter + 1 if frame > 100 and reward < 0 else 0
                if done: break
                # if neg_counter >= 25 or episode_return < 0:
                #     print("breaking {}".format(frame))
                #     break
                # Train
                if len(network.memory) >= args.batch_size:
                    network.train_on_batch()
                    trained_frames += 1
                frame += 1
            # RUN EPISODE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            episodes += 1
            print("episode: {}, trained_on: {} batches, return {}".format(episodes, trained_frames, episode_return))

            # Copy weights to fixed network
            if episodes % args.target_update_freq == 0 and len(network.memory) >= args.batch_size:
                network.copy_weights_to_fixed()
                dotsx.append(len(means_returns))
                dotsy.append(means_returns[-1])

            # Update epsilon ...
            epsilon = max(epsilon * args.epsilon_decay, args.epsilon_final)

            # Save stats ...
            returns.append(episode_return)
            means_returns.append(np.mean(np.asarray(returns)[-min(15, len(returns) - 1):]))
            log(log_name, episodes, means_returns)

            # Plot stats ...
            if args.renderable: plot_data(means_returns, dotsx, dotsy)

            # Save weights ...
            if len(returns) > 15:
                if np.average(returns[-15:]) > best_episode:
                    best_episode = np.average(returns[-15:])
                    network._model.save(args.model_name, include_optimizer=False)
                    print("Best so far: {}".format(best_episode))


if __name__ == "__main__":
    args = parser.parse_args([] if "file" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CarRacingSoftFS{}-v0".format(args.frame_skip)), args.seed,
                                     evaluate_for=15, report_each=1)
    main(env, args)
