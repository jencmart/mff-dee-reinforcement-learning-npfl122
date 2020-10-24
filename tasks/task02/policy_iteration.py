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

    @staticmethod
    def step(state, action):
        return [GridWorld._step(0.8, state, action),
                GridWorld._step(0.1, state, (action + 1) % 4),
                GridWorld._step(0.1, state, (action + 3) % 4)]

    @staticmethod
    def _step(probability, state, action):
        if state >= 5: state += 1
        x, y = state % 4, state // 4
        offset_x = -1 if action == 3 else action == 1
        offset_y = -1 if action == 0 else action == 2
        new_x, new_y = x + offset_x, y + offset_y
        if not(new_x >= 4 or new_x < 0  or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
            state = new_x + 4 * new_y
        if state >= 5: state -= 1
        return [probability, +1 if state == 10 else -100 if state == 6 else 0, state]


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")
parser.add_argument("--iterations", default=1, type=int, help="Number of iterations in policy evaluation step.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--steps", default=10, type=int, help="Number of policy evaluation/improvements to perform.")
# If you add more arguments, ReCodEx will keep them with your default values.

def calcualte_avg_from_p(rozdeleni_p, v, gamma):
    avg = 0
    for dynam in rozdeleni_p:
        prob, reward, s_new = dynam
        avg += prob * (reward + gamma*v[s_new])
    return avg

def calculate_argmax(arr):
    max = None
    max_idx = None
    for idx, i in enumerate(arr):
        if max is None or i > max:
            max = i
            max_idx = idx
    return max_idx

def main(args):
    gamma = 1
    # Start with zero value function and "go North" policy
    value_function = [0] * GridWorld.states
    policy = [0] * GridWorld.states  # policy je deterministicka

    for _ in range(args.steps):
        # >>> Evaluation <<<<
        for _ in range(args.iterations):
            value_function_new = []
            for state_idx in range(GridWorld.states):
                value_function_new.append(calcualte_avg_from_p(GridWorld.step(state_idx, policy[state_idx]), value_function, args.gamma))
            value_function = value_function_new

        # >>> Improvement <<
        for state_idx in range(GridWorld.states):
            action_rewards = []
            for action_idx in range(len(GridWorld.actions)):
                action_rewards.append(calcualte_avg_from_p(GridWorld.step(state_idx, action_idx), value_function, args.gamma))
            policy[state_idx] = calculate_argmax(action_rewards)

    return value_function, policy


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
