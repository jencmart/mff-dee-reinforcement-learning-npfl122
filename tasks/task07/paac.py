#!/usr/bin/env python3
import argparse
import os

# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import zipfile
import shutil
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=198, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0006, type=float, help="Learning rate.")
parser.add_argument("--workers", default=4, type=int, help="Number of parallel workers.")


class Network:
    def __init__(self, env, args):
        # TODO: Similarly to reinforce with baseline, define two models:
        #  - actor, which predicts distribution over the actions
        #  - critic, which predicts the value function

        # Use independent networks for both of them, each with
        # `args.hidden_layer_size` neurons in one ReLU hidden layer,
        # and train them using Adam with given `args.learning_rate`.

        input_layer = tf.keras.layers.Input(env.observation_space.shape[0])
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(input_layer)
        output_layer = tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax)(hidden_layer)
        self.actor = tf.keras.Model(input_layer, output_layer)
        self.actor.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        )
        input_layer = tf.keras.layers.Input(env.observation_space.shape[0])
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(input_layer)
        output_layer = tf.keras.layers.Dense(1)(hidden_layer)
        self.critic = tf.keras.Model(input_layer, output_layer)
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
        )

    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    # @tf.function
    def train(self, states, actions, returns):
        # TODO: Train the policy network using policy gradient theorem and the value network using MSE.
        self.critic.train_on_batch(states, returns)
        self.actor.train_on_batch(states, actions, sample_weight=returns - self.predict_values(states))

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states):
        return self.actor(states)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        # TODO: Return estimates of value function.
        result = self.critic(tf.convert_to_tensor([states], dtype=np.float32))
        return result[:, 0]


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def save_model(network, name):
    network.save(name, include_optimizer=False)
    zipf = zipfile.ZipFile(f'{name}.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir(f'{name}/', zipf)
    zipf.close()
    shutil.rmtree(name)


def load_model(name):
    with zipfile.ZipFile(name + ".zip", 'r') as zip_ref:
        zip_ref.extractall("./")
    return tf.keras.models.load_model(name)


def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation=False):
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Predict the action using the greedy policy
            actions = network.predict_actions(np.asarray([state]))[0]
            action = np.argmax(actions)
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.vector.AsyncVectorEnv([lambda: gym.make(env.spec.id)] * args.workers)
    states = vector_env.reset()
    best_so_far = 300
    all_returns = []
    training = not args.recodex
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Choose actions using network.predict_actions
            action_distributions = [network.predict_actions(np.asarray([state]))[0] for state in states]
            actions = [np.random.choice(env.action_space.n, p=action_dist) for action_dist in action_distributions]
            # compared to the actual paac alg here we perform only one action in each environment .. in the slides we
            # performed a series of actions in each env

            # TODO: Perform steps in the vectorized environment
            # each state is a vector of 4 numbers ... new_states is a matrix where each row is the quadruple of numbers
            # that describes the given state and we have as many rows as workers (as the environments)
            new_states, rewards, dones, _ = vector_env.step(actions)
            # TODO: Compute estimates of returns by one-step bootstrapping
            # episode returns are estimated as r + gamma * network.predict_values(...)
            episode_returns = np.zeros(len(new_states))

            for r, ns, d, i in zip(rewards, new_states, dones, range(len(new_states))):
                episode_returns[i] = r
                if not d:
                    episode_returns[i] += args.gamma * network.predict_values(ns)[0]

            # TODO: Train network using current states, chosen actions and estimated returns
            for ret, s, a in zip(episode_returns, states, actions):
                network.train(np.asarray([s]), np.asarray([a]), np.asarray([ret]))

            # need to make new states the old one
            states = new_states

        # Periodic evaluation
        for _ in range(args.evaluate_for):
            all_returns.append(evaluate_episode())

        if len(all_returns) > 200 and np.average(all_returns[-150:]) > best_so_far:
            best_so_far = np.average(all_returns[-150:])
            print("Best so far: {:.5}".format(best_so_far))
            save_model(network.actor, "paac_actor_model")
            save_model(network.critic, "paac_critic_model")

    network.actor = load_model('paac_actor_model')
    network.critic = load_model('paac_critic_model')
    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make(args.env), args.seed)

    main(env, args)
