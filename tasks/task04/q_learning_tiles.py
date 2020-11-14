#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers
# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=10, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=1, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
# must have alpha = 0.1, epsilon = 0.2, gamma = 1
# generally smaller epsilon tends to give better results having alpha=0.1, gamma=1
#   can consider epsilon 0.05
# also smaller learning rate works fine say 0.05 ... now after dividing alpha by the number of tiles alpha=0.1 works
# also larger number of tails should be compensated by the smaller learing rate
parser.add_argument("--alpha", default=0.15, type=float, help="Learning rate.")
parser.add_argument("--gamma", default=0.997, type=float, help="Discounting factor.")
parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")

parser.add_argument("--epsilon_final", default=0.001, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_decay_rate", default=0.9995, type=float, help="Number of tiles.")
parser.add_argument("--alpha_final", default=0.005, type=float, help="Final exploration factor.")
parser.add_argument("--alpha_decay_rate", default=0.99962, type=float, help="Number of tiles.")


def update_epsilon(epsilon, args):
    if epsilon > args.epsilon_final:
        return epsilon * args.epsilon_decay_rate
    else:
        return args.epsilon_final


def update_alpha(alpha, alpha_final):
    if alpha > alpha_final:
        return alpha * args.alpha_decay_rate
    else:
        return args.alpha_final


def main(env, args):
    # Fix random seed
    np.random.seed(args.seed)

    # environment is already created in the main main method :)

    weights_cnt = env.observation_space.nvec[-1]
    generator = np.random.RandomState(args.seed)

    # env.observation_space.nvec is a list of length args.tiles

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([weights_cnt, env.action_space.n])

    epsilon = args.epsilon
    alpha = args.alpha / args.tiles
    alpha_final = args.alpha_final / args.tiles

    training = not args.recodex
    returns = []

    while training:
        # Perform episode
        state, done = env.reset(), False
        # state is a list ... [20, 86, 176, 257, 338, 419, 500, 581]
        current_return = 0
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            # TODO: Choose an action based on epsilon greedy policy.
            action = np.argmax(np.sum(W[state], axis=0))
            if generator.uniform() < epsilon:
                action = generator.randint(env.action_space.n)

            next_state, reward, done, _ = env.step(action)
            current_return += reward

            # TODO: Update the action-value estimates
            update_weight = reward + args.gamma * np.max(np.sum(W[next_state], axis=0)) - np.sum(W[state, action])
            W[state, action] += alpha * update_weight * np.full(args.tiles, 1)

            state = next_state
        returns.append(current_return)
        if len(returns) > 5000 and np.average(returns[-1000:]) >= -105:
            print(f'Training finished with {np.average(returns[-1000:])}')
            break

        epsilon = update_epsilon(epsilon, args)
        alpha = update_alpha(alpha, alpha_final)

    if training:
        np.save('q_learning_tiles_weights.npy', W)
        print('saving weights')
    else:
        print('uploading weights')
        W = np.load('q_learning_tiles_weights.npy')

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            # TODO: Choose (greedy) action
            action = np.argmax(np.sum(W[state], axis=0))
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0"), tiles=args.tiles), args.seed)

    main(env, args)
