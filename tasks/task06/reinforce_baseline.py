#!/usr/bin/env python3
import argparse
import collections
import os

# 47b0acaf-eb3e-11e9-9ce9-00505601122b
# 8194b193-e909-11e9-9ce9-00505601122b


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=10, type=int, help="Number of episodes to train on.")
parser.add_argument("--episodes", default=600, type=int, help="Training episodes.")  # FIXED ! 500 <5min ;; 700=moc ;; 600?
parser.add_argument("--gamma", default=0.99, type=float, help="gamma for discount.")
parser.add_argument("--hidden_layer", default=238, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.003, type=float, help="Learning rate.")

parser.add_argument("--hidden_layer_baseline", default=220, type=int, help="Size of hidden layer.")
# 0.02 -- to much
# 0.01 -- ok
parser.add_argument("--learning_rate_baseline", default=0.003, type=float, help="Learning rate.")


# class Network:
#     def __init__(self, env, args):
#         # TODO: Create a suitable model.
#         #
#         # Apart from the model defined in `reinforce`, define also another
#         # model for computing baseline (with one output, using a dense layer
#         # without activation).
#         #
#         # Using Adam optimizer with given `args.learning_rate` for both models
#         # is a good default.
#         raise NotImplementedError()
#
#     # TODO: Define a training method. Note that we need to use @tf.function for
#     # efficiency (using `train_on_batch` on extremely small batches/networks
#     # has considerable overhead).
#     @tf.function(experimental_relax_shapes=True)
#     def train(self, states, actions, returns):
#         # You should:
#         # - compute the predicted baseline using the baseline model
#         # - train the policy model, using `returns - predicted_baseline` as
#         #   advantage estimate
#         # - train the baseline model to predict `returns`
#         raise NotImplementedError()
#
#     # Predict method, again with manual @tf.function for efficiency.
#     @tf.function
#     def predict(self, states):
#         return self._model(states)

class BaselineNetwork:
    d1 = []
    d2 = []
    def __init__(self, env, args):
        inputs = tf.keras.layers.Input(shape=env.observation_space.shape)
        x = tf.keras.layers.Dense(units=args.hidden_layer_baseline, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(units=1)(x)  # single output
        self.model = tf.keras.Model([inputs], [outputs], name="baseline_model")

        o = tf.optimizers.Adam(lr=args.learning_rate_baseline)
        self.model.compile(optimizer=o, loss='mse')

    def train(self, states, returns):
        states, returns = np.array(states, np.float32), np.array(returns, np.float32)
        # States  [ batch , 4 ]
        # returns [ batch , 1 ]

        self.model.train_on_batch(x=states, y=returns)

    def predict(self, states):
        states = np.array(states, np.float32)
        results = self.model.predict_on_batch(x=states)
        # results = results.numpy()
        results = results[:, 0]
        return results


class Network:
    d1 = []
    d2 = []
    def __init__(self, env, args):

        # The inputs have shape `env.state_shape`,
        # and the model should produce probabilities of `env.actions` actions.
        # You can use for example one hidden layer with `args.hidden_layer` and non-linear activation.
        inputs = tf.keras.layers.Input(shape=env.observation_space.shape)
        x = tf.keras.layers.Dense(units=args.hidden_layer)(inputs)
        # dropout
        # x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.activations.relu(x)
        outputs = tf.keras.layers.Dense(units=env.action_space.n, activation=tf.nn.softmax)(x)
        self.model = tf.keras.Model([inputs], [outputs], name="reinforce_model")

        o = tf.optimizers.Adam(lr=args.learning_rate)
        self.model.compile(optimizer=o, loss='mse')

        self.baseline_network = BaselineNetwork(env, args)

    def train(self, states, actions, returns):
        states, actions, returns = np.array(states, np.float32), np.array(actions, np.int32), np.array(returns,
                                                                                                       np.float32)
        # States  [ batch , 4 ]
        # Actions [ batch , 2 ]
        # returns [ batch , 1 ]

        # todo: first train the baseline
        self.baseline_network.train(states, returns)

        onehot_actions = np.zeros((actions.size, actions.max() + 1), dtype=np.int32)
        onehot_actions[np.arange(actions.size), actions] = 1

        # todo: predict baseline using baseline network
        baseline = self.baseline_network.predict(states)

        returns -= baseline
        self.model.train_on_batch(x=states, y=onehot_actions, sample_weight=returns)

    def predict(self, states):
        states = np.array(states, np.float32)
        # Predict distribution over actions for the given input states
        # using the `predict_on_batch` method and calling `.numpy()` on the result to return a NumPy array.
        results = self.model.predict_on_batch(x=states)
        # results = results.numpy()
        return results

def calculate_returns(rewards, gamma=0.99, subs_mean=False):
    # apply discount [1, 0.99, 0.98, 0.97]
    returns = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
    # cumsum from back:
    # [1,1,1,1,1] -> [5,4,3,2,1]
    # [1, 0.99, 0.98, 0.97]   --> [5, 4, 3.1, 2.2, 1.3]
    returns = np.cumsum(returns[::-1])[::-1]
    if subs_mean:
        returns -= returns.mean()
    return returns

def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    possible_actions = list(range(env.action_space.n))

    # Training
    episode_returns = []
    for _ in range(args.episodes // args.batch_size):
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
        avg_100 = np.mean(np.asarray(episode_returns[-100:]))
        # print(avg_100)
        if len(episode_returns) > 100 and avg_100 > 499:
            break

        network.train(batch_states, batch_actions, batch_returns)
    n1 = network.model.get_weights()
    n2 = network.baseline_network.model.get_weights()
    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPole-v1"), args.seed)

    main(env, args)
