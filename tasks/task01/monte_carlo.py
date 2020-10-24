#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.13, type=float, help="Exploration factor.")

def main(env, args):
    # Fix random seed
    np.random.seed(args.seed)
    # may be stochastic in states
    # may be stochastic in rewards (often depends on state...)
    # ...distribution over (state,reward) states
    # reward X return
    # reward ... we get it at each time step
    # return ... sum of all rewards ;; we have discount factor ;; gamma ;; ofter 1
    # in multi armed bandis you have only 1 state (it does not change ...)
    # we need to estime action in each state ....
    # DONE
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    C = np.zeros([env.observation_space.n, env.action_space.n])

    for _ in range(args.episodes):
        # TODO: Perform episode
        #  == collecting states, actions, rewards
        states = []
        actions = []
        rewards = []

        # G <- returns
        state, done = env.reset(), False
        while not done:

            # render ...
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()
            # TODO: Compute `action` using epsilon-greedy policy.
            if np.random.uniform(0, 1) > args.epsilon:
                action = np.argmax(Q[state, :])
            else:
                action = np.random.choice(env.action_space.n)

            # Do the step...
            next_state, reward, done, _ = env.step(action)
            actions.append(action)
            states.append(state)
            rewards.append(reward)

            state = next_state
        # TODO: Compute returns from the recieved rewards and update Q and C.
        G = 0
        states.reverse()
        actions.reverse()
        rewards.reverse()
        for s, a, r in zip(states, actions, rewards):
            G += r
            C[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / C[s, a]

    # >>>>>>>>> Final evaluation <<<<<<<<<<<<<
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose greedy action
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed)

    main(env, args)
