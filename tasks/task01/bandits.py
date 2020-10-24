#!/usr/bin/env python3
import argparse
import sys
import numpy as np

# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b


class MultiArmedBandits():
    def __init__(self, bandits, seed=None):
        self._generator = np.random.RandomState(seed)
        self._bandits = [None] * bandits
        self.reset()

    def reset(self):
        for i in range(len(self._bandits)):
            self._bandits[i] = self._generator.normal(0., 1.)

    def step(self, action):
        return self._generator.normal(self._bandits[action], 1.)

    def greedy(self, epsilon):
        return self._generator.uniform() >= epsilon


parser = argparse.ArgumentParser()

# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use or 0 for averaging.")
parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")
parser.add_argument("--episodes", default=100, type=int, help="Episodes to perform.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
parser.add_argument("--initial", default=0, type=float, help="Initial estimation of values.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")


# If you add more arguments, ReCodEx will keep them with your default values.
def main(env, args):
    # TODO: Initialize the estimates, to `args.initial`.
    curr_estim = np.full(args.bandits, args.initial)
    counts = np.zeros(args.bandits)
    rewards = 0
    for step in range(args.episode_length):
        # TODO: Select an action according to `env.greedy(args.epsilon)`.
        if env.greedy(args.epsilon):
            action = np.argmax(curr_estim)
        else:
            action = np.random.choice(args.bandits)

        # Perform it
        reward = env.step(action)
        rewards += reward
        counts[action] += 1

        # TODO: Update parameters, either using averaging (when `args.alpha` == 0)
        # or by an update with a learning rate of `args.alpha`.

        if args.alpha == 0:
            curr_estim[action] += 1/counts[action]*(reward - curr_estim[action])
        else:
            curr_estim[action] += args.alpha * (reward - curr_estim[action])

    return rewards / args.episode_length


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    banditsEnv = MultiArmedBandits(args.bandits, seed=args.seed)

    # Fix random seed
    np.random.seed(args.seed)

    returns = []
    for _ in range(args.episodes):
        returns.append(main(banditsEnv, args))

    # Print the mean and std
    print("{:.2f} {:.2f}".format(np.mean(returns), np.std(returns)))
