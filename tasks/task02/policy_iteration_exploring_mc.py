#!/usr/bin/env python3
import argparse

import numpy as np

class GridWorld:
    # States in the gridworld are the following:
    # 0 1 2 3
    # 4 x 5 6
    # 7 8 9 10

    # The rewards are +1 in state 10 and -100 in state 6

    # Actions are ↑ → ↓ ←; with probability 80% they are performed as requested,
    # with 10% move 90° CCW is performed, with 10% move 90° CW is performed.
    states = 11

    actions = ["↑", "→", "↓", "←"]

    def __init__(self, seed):
        self._generator = np.random.RandomState(seed)

    def step(self, state, action):
        probability = self._generator.uniform()
        if probability <= 0.8: return self._step(state, action)
        if probability <= 0.9: return self._step(state, (action + 1) % 4)
        return self._step(state, (action + 3) % 4)

    @staticmethod
    def _step(state, action):
        if state >= 5: state += 1
        x, y = state % 4, state // 4
        offset_x = -1 if action == 3 else action == 1
        offset_y = -1 if action == 0 else action == 2
        new_x, new_y = x + offset_x, y + offset_y
        if not(new_x >= 4 or new_x < 0 or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
            state = new_x + 4 * new_y
        if state >= 5: state -= 1
        return (+1 if state == 10 else -100 if state == 6 else 0, state)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")
parser.add_argument("--mc_length", default=100, type=int, help="Monte Carlo simulation episode length")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--steps", default=10, type=int, help="Number of policy evaluation/improvements to perform.")
# If you add more arguments, ReCodEx will keep them with your default values.


def monte_carlo_simulation(gamma, env, policy, steps, first_state, first_action, Q, C):
    states, actions, rewards = [], [], []
    state = first_state
    action = first_action

    # G <- returns
    for _ in range(steps):
        # Do the step...
        reward, next_state = env.step(state, action)

        # Append
        actions.append(action)
        states.append(state)  # last state will be ommited and that is ok
        rewards.append(reward)

        state = next_state
        action = policy[state]

    # Calculate retuns ..
    returns = 0
    for s, a, r in zip(reversed(states), reversed(actions), reversed(rewards)):
        returns = returns * gamma + r
    C[first_state, first_action] += 1
    Q[first_state, first_action] += (returns - Q[first_state, first_action]) / C[first_state, first_action]


def main(args):
    env = GridWorld(args.seed)
    Q, C = np.zeros((env.states, len(env.actions))), np.zeros((env.states, len(env.actions)))
    policy = np.zeros(env.states, np.int32)
    for _ in range(args.steps):
        # >>> EVALUATION <<<
        for s in range(env.states):
            for a in range(len(env.actions)):
                monte_carlo_simulation(gamma=args.gamma, env=env, policy=policy, steps=args.mc_length, first_state=s, first_action=a, Q=Q, C=C)
        # >>> IMPROVEMENT <<
        policy = [np.argmax(Q[s]) for s in range(GridWorld.states)]
        v_function = [Q[s, policy[s]] for s in range(GridWorld.states)]
    return v_function, policy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    value_function, policy = main(args)

    # Print results
    for l in range(3):
        for c in range(4):
            state = l * 4 + c
            if state >= 5: state -= 1
            print("        " if l == 1 and c == 1 else "{:-8.2f}".format(value_function[state]), end="")
            print(" " if l == 1 and c == 1 else GridWorld.actions[policy[state]], end="")
        print()
