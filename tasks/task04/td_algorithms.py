#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor gamma.")
parser.add_argument("--mode", default="tree_backup", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy; use greedy as target")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.


# TODO - DONE ... funguje
# s_t, a_t, r_t+1, s_t+1, a_t+1
def tree_n_step(episode, gamma, use_off_policy, target_policy_pi, Q, done):
    # s_from, a, r, s_next
    s_from = episode[0][0]
    a_prob = episode[0][2]  # prob. w.r.t. behavior policy
    reward = episode[0][3]
    s_next = episode[0][4]
    a_next = episode[0][5]  # true next action (selected by behavior policy)

    if len(episode) == 1:
        end_val = np.dot(target_policy_pi[s_next], Q[s_next])  # np.max(Q[s_next]) if use_off_policy else
        if done:  # if we are done return of s_next will obviously be 0
            return reward
        else:  # expectation over all actions
            return reward + gamma*end_val

    # expectation over not-selected-action
    selected_a = a_next

    # E_pi over not selected actions
    E_pi_notA_S = np.dot(np.delete(target_policy_pi[s_next], selected_a), np.delete(Q[s_next], selected_a))
    # selected action
    pi_A_S = target_policy_pi[s_next, selected_a]

    G_t_n = reward \
            + gamma * E_pi_notA_S \
            + gamma * pi_A_S * tree_n_step(episode[1:], gamma, use_off_policy, target_policy_pi, Q, done)

    return G_t_n


# todo - nefunguje n>1 OFF policy ...
def n_step_expected_sarsa(episode, gamma, use_off_policy, target_policy_pi, Q, done):
    reward = episode[0][3]
    s_next = episode[0][4]  # prob. w.r.t. behavior policy

    if len(episode) == 1:
        end_val = np.dot(target_policy_pi[s_next], Q[s_next])  # np.max(Q[s_next]) if use_off_policy else
        if done:  # if we are done return of s_next will obviously be 0
            return reward
        else:  # expectation over all actions
            return reward + gamma * end_val

    G_t_n = reward + gamma * n_step_expected_sarsa(episode[1:], gamma, use_off_policy, target_policy_pi, Q, done)
    return G_t_n

# todo - nefunguje n>1 OFF policy ...
def n_step_srsa(episode, gamma, use_off_policy, target_policy_pi, Q, done):
    reward = episode[0][3]
    s_next = episode[0][4]  # prob. w.r.t. behavior policy
    a_next = episode[0][5]  # next action prob w.r.t. behavior policy

    if len(episode) == 1:
        if done:  # if we are done return of s_next will obviously be 0
            return reward
        if use_off_policy:
            act = np.argmax(target_policy_pi[s_next])  # arg-max z target policy ...
            end_val = Q[s_next, act]  # stejne jako max(Q[s_next]) .. alespon tedy pro n=1
        else:
            end_val = Q[s_next, a_next]
        return reward + gamma*end_val

    G_t_n = reward + gamma * n_step_srsa(episode[1:], gamma, use_off_policy, target_policy_pi, Q, done)
    return G_t_n



def update_last_n_steps(use_off_policy, algo_type, episode, Q, target_policy_pi, n, gamma, alpha, done=False):
    state = episode[0][0]
    action = episode[0][1]
    sampling = False

    if algo_type == "tree_backup":
        G_t_n = tree_n_step(episode, gamma, use_off_policy, target_policy_pi, Q, done)

    elif algo_type == "expected_sarsa":
        if use_off_policy and n > 1:
            sampling = True
        G_t_n = n_step_expected_sarsa(episode, gamma, use_off_policy, target_policy_pi, Q, done)

    elif algo_type == "sarsa":
        if use_off_policy:
            sampling = True
        G_t_n = n_step_srsa(episode, gamma, use_off_policy, target_policy_pi, Q, done)
    else:
        raise NotImplementedError()

    rho = 1
    if sampling:

        # todo - z bude fungovat off policy n step sarsa
        # tak expected sarsa konci jeste o jeden index driv ...
        if algo_type == "expected_sarsa":
            tmp = episode[:-1]
        else:
            tmp = episode
        for idx, e in enumerate(tmp):
            if done and idx+1 == len(episode):
                break
            [s, a, ap, r, sn, an, anp] = e
            rho *= target_policy_pi[sn, an] / anp

    Q[state, action] += alpha * (rho if sampling else 1) * (G_t_n - Q[state, action])



def main(args):
    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("Taxi-v3"), seed=args.seed, report_each=100)
    # Fix random seed and create a generator
    generator = np.random.RandomState(args.seed)

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(args.episodes):
        next_state, done = env.reset(), False

        # Generate episode and update Q using the given TD method
        next_action = np.argmax(Q[next_state]) if generator.uniform() >= args.epsilon else env.action_space.sample()
        next_action_prob = args.epsilon / env.action_space.n + (1 - args.epsilon) * (next_action == np.argmax(Q[next_state]))
        episode =[]
        while not done:
            action, action_prob, state = next_action, next_action_prob, next_state
            next_state, reward, done, _ = env.step(action)
            if not done:
                next_action = np.argmax(Q[next_state]) if generator.uniform() >= args.epsilon else env.action_space.sample()
                next_action_prob = args.epsilon / env.action_space.n + (1 - args.epsilon) * (next_action == np.argmax(Q[next_state]))

            target_policy = np.eye(env.action_space.n)[np.argmax(Q, axis=1)]
            if not args.off_policy:
                target_policy = (1 - args.epsilon) * target_policy + args.epsilon / env.action_space.n * np.ones_like(target_policy)
            # todo - do the update ...
            episode.append([state, action, action_prob, reward, next_state, next_action, next_action_prob])
            if len(episode) == args.n:
                update_last_n_steps(args.off_policy, args.mode, episode, Q, target_policy, args.n, args.gamma, args.alpha, done)
                episode = episode[1:]  # remove the oldest one ..

        # todo - do the update ...
        while len(episode) >= 1:
            update_last_n_steps(args.off_policy, args.mode, episode, Q, target_policy, args.n, args.gamma, args.alpha, done)
            episode = episode[1:]

    return Q


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
