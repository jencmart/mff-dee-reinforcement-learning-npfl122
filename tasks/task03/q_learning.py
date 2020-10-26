#!/usr/bin/env python3
import argparse
import gym
import numpy as np
import wrappers
# 576 sates
# 3 actions

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=100, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")  ## todo -- set this
parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.") ## todo -- set this
parser.add_argument("--gamma", default=0.9, type=float, help="Discounting factor.")
parser.add_argument("--training_alg", default="2Q", type=str, help="Training type.")

# gamma 1
# The mean 100-episode return after evaluation -201.21 +-22.93
# gamma 0.75
# The mean 100-episode return after evaluation -157.86 +-16.82

def render(render_each, episode):
    if render_each and episode >= render_each and episode > 0 and episode % render_each == 0:
        env.render()


def epsilon_greedy(epsilon, estimation, actions):
    if np.random.uniform() < epsilon:
        return np.random.randint(actions)
    else:
        return np.argmax(estimation)


def double_q_learning(env, args):
    epsilon = args.epsilon
    alpha = args.alpha
    print("Double Q Learning")
    training = True
    Q1 = np.zeros([env.observation_space.n, env.action_space.n])
    Q2 = np.zeros([env.observation_space.n, env.action_space.n])
    all_returns = []

    episode = 0
    while training:
        episode += 1
        if len(all_returns) > 1000:
            if np.abs(np.mean(np.asarray(all_returns[-100:]))) < 145:
                break
        # if episode == 20000:
        #     break
        if episode % 1000 == 0:
            epsilon /= 2
            # alpha = 1 / (episode / 300)
            print("epsilon " + str(epsilon))
            print("aplha " + str(alpha))

        # Perform episode
        episode_return = 0
        s, done = env.reset(), False
        while not done:
            a = epsilon_greedy(epsilon, Q1[s]+Q2[s], env.action_space.n)
            # Perform step
            next_s, r, done, _ = env.step(a)
            episode_return += r
            # Double Q Learning
            if np.random.uniform() < 0.5:
                Q1[s, a] += alpha*(r + args.gamma * Q2[next_s, np.argmax(Q1[next_s])] - Q1[s, a])
            else:
                Q2[s, a] += alpha*(r + args.gamma * Q1[next_s, np.argmax(Q2[next_s])] - Q2[s, a])
            s = next_s
        all_returns.append(episode_return)
    pi = [np.argmax(Q1[s] + Q2[s]) for s in range(env.observation_space.n)]
    return pi


def q_learning(env,args):
    print("Q Learning")
    training = True
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    rwrd = 1000
    while training:
        # Perform episode
        s, done = env.reset(), False
        while not done:
            render(args.render_each, env.episode)
            # Select action
            a = epsilon_greedy(args.epsilon, Q[s], env.action_space.n)
            # Perform step
            next_s, r, done, _ = env.step(a)
            rwrd -= 1
            # Q learning
            # Q[s, a]+=args.alpha * (r + args.gamma * np.argmax(Q[next_s]) - Q[s, a])
            # sarsa
            Q[s, a] += args.alpha * (r + args.gamma * Q[next_s, np.argmax(Q[next_s])] - Q[s, a])
            s = next_s
        print("rwrd: " + str(rwrd))
    pi = [np.argmax(Q[s]) for s in range(env.observation_space.n)]
    return pi


def main(env, args):
    # Fix random seed
    np.random.seed(args.seed)

    if args.training_alg == "Q":
        pi = q_learning(env, args)
    elif args.training_alg == "2Q":
        pi = double_q_learning(env, args)
    else:
        raise BaseException("Unknown training " + args.training_alg)

    # Final evaluation
    print("Evaluation...")
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            action = pi[state]
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0")), args.seed)
    main(env, args)
