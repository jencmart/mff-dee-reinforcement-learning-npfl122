#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=20, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")  # Can be fixed ; 0.1 is fine ... 0.2 is also fine
parser.add_argument("--aplha_epsiode_decay", default=0.0003, type=float, help="Decay of learning rate.")  # Can be fixed ; 0.1 is fine ...
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")  # Start high and decay ... OK
parser.add_argument("--epsilon_epsiode_decay", default=0.001, type=float, help="Decay of exploration.")
parser.add_argument("--gamma", default=0.75, type=float, help="Discounting factor.")  # I dunno ...
parser.add_argument("--training_alg", default="TB", type=str, help="Training type.")  # Now not used ...

# 0: Do nothing
# 1: Fire left thruster
# 2: Fire main thruster
# 3: Fire right thruster

# This class is used for the training ... It can do whatever TD you want
# 1-step n-step on-policy off-policy with/without expectation ... every combination is possible through configuration
class ReinforceTD:
    def __init__(self, env, args):
        # Default init ...
        self.env = env
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.C = np.zeros([env.observation_space.n, env.action_space.n])
        self.cnt_states = env.observation_space.n
        self.cnt_actions = env.action_space.n
        self.episode_done = False
        self.episode_return = 0

        # Stopping criterion...
        self.returns_of_episodes = []
        self.avg_over_episodes = 100
        self.best_mu = None  # best average over self.avg_over_episodes so far ...
        self.best_Q = None  # Q corresponding to the best mu ...
        self.target_avg = 200   # we want to reach this target average over self.avg_over_episodes
        self.max_episodes = 100000  # but we won't continue if we have reached this many episodes ...

        # >>>>>>>> HYPER PARAMETERS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.n = 10  # n-step TD  4 works great
        self.gamma = args.gamma  # Constant discount factor ...

        # >>> BEHAVIOR POLICY
        # Start with whatever behavior policy you want (ideally uniform or proportional to s.t. you explore a lot...
        initial_policy = ["epsilon_greedy", "uniform",  "greedy", "proportional"]
        self.behavior_policy = initial_policy[3]
        # But after some time, we always switch to decaying e-greedy policy
        self.change_to_decaying_epsilon_greedy_in_episode = 1000
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_epsiode_decay

        # >>> LEARNING RATE
        self.alpha = args.alpha
        self.aplha_epsiode_decay = args.aplha_epsiode_decay
        self.do_learning_decay = True
        # >>>>>>>> HYPER PARAMETERS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def get_greedy_pi_action(self, state):
        return np.argmax(self.Q[state])

    def init_train_episode(self):
        self.episode_done = False
        self.episode_return = 0
        s = self.env.reset()
        return s

    def select_next_action(self, s):
        if self.behavior_policy == "epsilon_greedy":
            return epsilon_greedy(self.epsilon, self.Q[s], self.cnt_actions)
        elif self.behavior_policy == "greedy":
            return np.argmax(self.Q[s])
        elif self.behavior_policy == "uniform":
            return np.random.randint(0, self.cnt_actions)
        elif self.behavior_policy == "proportional":
            tmp = self.Q[s]
            tmp = tmp + np.abs(np.min(tmp))
            if np.sum(tmp) == 0:  # prevent first state ...
                return np.random.randint(0, self.cnt_actions)
            prob = tmp / np.sum(tmp)
            return np.random.choice(np.arange(self.cnt_actions), p=prob)
        else:
            raise BaseException("Unknown behavioral policy {}".format(self.behavior_policy))


    def perform_step(self, a):
        render(1000, self.env)
        next_s, r, self.episode_done, _ = self.env.step(a)
        self.episode_return += r
        return next_s, r

    # def __expectation_of_Q_wrt_pi_without_a(self, next_s, true_a):
    #     greedy_a = self.get_greedy_pi_action(next_s)
    #     if true_a == greedy_a:
    #         return 0
    #     return 1

    def __expected_Q_wrt_pi(self, next_state):
        # 0*akce +  0*akce +  0*akce + 1*greedy_akce
        return np.max(self.Q[next_state])

    # def __prob_of_wrt_pi(self, state, action):
    #     if self.get_greedy_pi_action(state) == action:
    #         return 1
    #     else:
    #         return 0

    def __recursive_n_step(self, episode):
        # s_from, a, r, s_next
        s_from = episode[0][0]
        action = episode[0][1]
        reward = episode[0][2]
        s_next = episode[0][3]
        if len(episode) == 1:
            return reward + self.__expected_Q_wrt_pi(s_next)
        # G_t_n = reward + self.args.gamma * self.expected_Q_wrt_pi(action, s_next) + self.args.gamma * self.__prob_of_wrt_pi(s_next, action) * self.__recursive_n_step(episode[1:])
        G_t_n = reward + self.gamma * self.__recursive_n_step(episode[1:])
        return G_t_n

    # def __update_last_n_steps(self, episode):
    #     if len(episode) >= self.n:
    #         G = 0
    #         s_start = episode[0][0]
    #         a_start = episode[0][1]
    #         for s, true_a, r in reversed(episode):
    #             E_pi = self.__expectation_of_Q_wrt_pi_without_a(s, true_a)
    #             G = r + self.args.gamma*E_pi + self.args.gamma * G
    #         self.Q[s_start, a_start] += args.alpha * (G - self.Q[s_start, a_start])
    #         return episode[1:]
    #     return episode

    def update(self, episode_list, last=False):
        if last or len(episode_list) >= self.n:
            G_t_n = self.__recursive_n_step(episode_list)
            state = episode_list[0][0]
            action = episode_list[0][1]
            self.Q[state, action] += args.alpha * (G_t_n - self.Q[state, action])
            return episode_list[1:]

        return episode_list

    def is_episode_end(self):
        return self.episode_done

    def at_episode_end(self):
        self.returns_of_episodes.append(self.episode_return)
        if self.behavior_policy == "epsilon_greedy":
            self.epsilon = self.epsilon*(1 - self.epsilon_decay)
            if self.do_learning_decay:
                self.alpha = self.alpha*(1 - self.aplha_epsiode_decay)
                self.alpha = max(self.alpha, 0.0001)
        self.episode_return = 0

        if len(self.returns_of_episodes) == self.change_to_decaying_epsilon_greedy_in_episode:
            self.behavior_policy = "epsilon_greedy"

    def is_train_end(self):
        mu = np.average(np.asarray(self.returns_of_episodes[-self.avg_over_episodes:]))
        if not self.best_mu or mu > self.best_mu:
            self.best_mu = mu
            self.best_Q = np.copy(self.Q)
        if len(self.returns_of_episodes) >= self.avg_over_episodes:
            if mu > self.target_avg or len(self.returns_of_episodes) > self.max_episodes:
                return True
        return False

    def __write_to_file(self, greedy_pi, mean):
        with open("pi.txt", 'a') as f:
            f.write(str(mean) + ",")
            for idx, p in enumerate(greedy_pi):
                f.write(str(p))
                if idx + 1 == len(greedy_pi):
                    f.write("\n")
                else:
                    f.write(",")

    def at_train_end(self):
        # do it from the best Q
        greedy_pi = [np.argmax(self.best_Q[s]) for s in range(self.cnt_states)]  # Akce a
        #     # self.V = [self.Q[s, self.greedy_pi[s]] for s in range(self.cnt_states)] # hodnoceni ...
        self.__write_to_file(greedy_pi, self.best_mu)
        print("Training done....")
        print("Episode: " + str(len(self.returns_of_episodes)) + ", mean 100-episode return: " + str(self.best_mu))
        print("epsilon: {}".format(self.epsilon))
        print("alpha: {}".format(self.alpha))
        return greedy_pi


def render(render_each, env):
    if render_each == 0 or env.episode < render_each:
        return
    if env.episode % render_each == 0:  # render ech ith episode ..
        env.render()


def epsilon_greedy(epsilon, estimation, actions):
    if np.random.uniform() < epsilon:
        return np.random.randint(actions)
    else:
        return np.argmax(estimation)

def reinforce_loop(obj):
    episode_cnt = 1
    while True:
        s = obj.init_train_episode()
        episode = []
        # print("Now comes episode: {}".format(episode_cnt))
        while True:
            a = obj.select_next_action(s)  # fast ...
            s_next, r = obj.perform_step(a)  # fast ...
            episode.append((s, a, r, s_next))  # fast ...
            episode = obj.update(episode)  # fast ...
            s = s_next
            if obj.is_episode_end():
                break
        while len(episode) >= 1:
            episode = obj.update(episode, last=True)

        episode_cnt += 1
        obj.at_episode_end()
        if obj.is_train_end():
            pi = obj.at_train_end()
            return pi

def tree_backup(env, args, n=4):
    print("Tree Backup ")
    all_returns = []

    def update_last_n_steps():
        # we can see that for greedy(argmax) policy pi  [what I have here] it is the same as Q learning ...
        # E_wrt_pi = Q[next_s, np.argmax(Q[next_s])]
        # G = r + args.gamma * E_wrt_pi
        # Q[s, a] += args.alpha * (G - Q[s, a])
        G = 0
        s_start = episode[0][0]
        a_start = episode[0][1]
        for s, a, r in reversed(episode):
            aa = np.argmax(Q[s])
            exepct_no_a = Q[s, aa] if a != aa else 0
            G = r + args.gamma*exepct_no_a + args.gamma * G
        Q[s_start, a_start] += args.alpha * (G - Q[s_start, a_start])

    training = True
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    tot_episode = 0
    while training:
        if len(all_returns) > 1000:
            mu = np.mean(np.asarray(all_returns[-100:]))
            if mu > 10:
                pi = [np.argmax(Q[s]) for s in range(env.observation_space.n)]
                with open("pi.txt", 'a') as f:
                    print(mu)
                    print(tot_episode)
                    for idx, p in enumerate(pi):
                        f.write(str(p))
                        if idx + 1 == len(pi):
                            f.write("\n")
                        else:
                         f.write(",")
                break

        # Perform episode
        s, done = env.reset(), False
        tot_episode += 1
        episode = []
        episode_return = 0
        while not done:
            if tot_episode > 1 and tot_episode % 100 == 0:
                render(args.render_each, env)
            a = epsilon_greedy(args.epsilon, Q[s], env.action_space.n)
            next_s, r, done, _ = env.step(a)
            episode_return += r
            episode.append((s, a, r))
            # Update
            if len(episode) == n:
                update_last_n_steps()
                episode = episode[1:]  # remove the oldest one ..
            s = next_s
        # update last few shits ...
        while len(episode) >= 1:
            update_last_n_steps()
            episode = episode[1:]
        all_returns.append(episode_return)
        if tot_episode == -100:
            expert_traject = False
            tot_episode = 0
            print("exiting expert trajectory ..")
    pi = [np.argmax(Q[s]) for s in range(env.observation_space.n)]
    return pi


def n_step_sars(env, args, n=4):
    print("N step sarsa SARSA")
    training = True
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    all_returns = []

    def update_last_n_steps():
        G = 0
        s_start = episode[0][0]
        a_start = episode[0][1]
        for s, a, r in reversed(episode):
            G = G * args.gamma + r
        Q[s_start, a_start] += args.alpha * (G - Q[s_start, a_start])
    tot_episode = 0
    while training:
        if len(all_returns) > 1000:
            if np.mean(np.asarray(all_returns[-100:])) > -150:
                pi = [np.argmax(Q[s]) for s in range(env.observation_space.n)]
                with open("pi.txt", 'a') as f:
                    f.write(pi)
                exit(1)
                break

        tot_episode += 1
        # Perform episode
        s, done = env.reset(), False

        episode = []
        while not done:
            if env.episode > 1 and tot_episode % 100 ==0:
                render(args.render_each, env)

            a = epsilon_greedy(args.epsilon, Q[s], env.action_space.n)
            next_state, reward, done, _ = env.step(a)
            episode.append((s, a, reward))
            # Do the update
            if len(episode) == n:
                update_last_n_steps()
                episode = episode[1:]  # remove the oldest one ..
            s = next_state

        # update last few shits ...
        while len(episode) >= 1:
            update_last_n_steps()
            episode = episode[1:]

    pi = [np.argmax(Q[s]) for s in range(env.observation_space.n)]
    return pi

def n_step_expected_sars(env, args):
    pass

def n_step_off_policy_sarsa(emv, args):
    pass

# Expected Sars is OFF Policy ; it is generalization of Q-learning
def expected_sarsa(env, args):
    print("Expected SARSA")
    training = True
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    while training:
        # Perform episode
        s, done = env.reset(), False
        while not done:
            render(args.render_each, env)
            # Select action
            a = epsilon_greedy(args.epsilon, Q[s], env.action_space.n)
            # Perform step
            next_s, r, done, _ = env.step(a)
            # we can see that for greedy(argmax) policy pi  [what I have here] it is the same as Q learning ...
            E_wrt_pi = Q[next_s, np.argmax(Q[next_s])]
            G = r + args.gamma * E_wrt_pi
            Q[s, a] += args.alpha * (G - Q[s, a])
            s = next_s
    pi = [np.argmax(Q[s]) for s in range(env.observation_space.n)]
    return pi

def q_learning(env,args):
    print("Q Learning")
    training = True
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    while training:
        # Perform episode
        s, done = env.reset(), False

        # SARSA
        # a = epsilon_greedy(args.epsilon, Q[s], env.action_space.n)
        # while not done:
        #     next_s, r, done, _ = env.step(a)
        #     a_estim = epsilon_greedy(args.epsilon, Q[next_s], env.action_space.n)
        #     v(St+1) = max a  (S[s+t1, a])
        #     Estimate using ε-greedy w.r.t. Q
        #     Q[s, a] += args.alpha * (r + args.gamma * Q[next_s, a_estim] - Q[s, a])
        #     Take ε-greedy w.r.t. Q
        #     a = a_estim
        #     s = next_s

        while not done:
            render(args.render_each, env)
            # Take ε-greedy w.r.t. Q .... behavior policy
            a = epsilon_greedy(args.epsilon, Q[s], env.action_space.n)
            next_s, r, done, _ = env.step(a)
            # But estimate using Greedy w.r.t. Q .... target policy
            a_estim = np.argmax(Q[next_s])
            Q[s, a] += args.alpha * (r + args.gamma * Q[next_s, a_estim] - Q[s, a])
            s = next_s
    pi = [np.argmax(Q[s]) for s in range(env.observation_space.n)]
    return pi

# Q learning is OFF Policy
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


def load_pi(path="pi.txt"):
    pi_vals = []
    pis = []

    with open(path) as f:
        for line in f:
            arr = line.split(',')
            pi_vals.append(arr[0])
            tmp = arr[1:]
            for i in range(0, len(tmp)):
                tmp[i] = int(tmp[i])
            pis.append(tmp)

    best_pi_idx = int(np.argmax(np.asarray(pi_vals)))

    print("Loaded pi with mean 100-episode return {}".format(pi_vals[best_pi_idx]))
    return pis[best_pi_idx]


def main(env, args):
    wwrap = ReinforceTD(env, args)
    if not args.recodex:
        pi = reinforce_loop(wwrap)
    else:
        pi = load_pi("pi.txt")

    # if not args.recodex:
    #     # Fix random seed
    #     np.random.seed(args.seed)
    #     if args.training_alg == "TB":
    #         pi = tree_backup(env, args)
    #     elif args.training_alg == "NSS":
    #         pi = n_step_sars(env, args)
    #     elif args.training_alg == "2Q":
    #         pi = double_q_learning(env, args)
    #     else:
    #         raise BaseException("Unknown training " + args.training_alg)
    # else:
    #     pi = load_pi("pi.txt")

    # Final evaluation
    print("Evaluation...")
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            render(args.render_each, env)  # pokud 20. epizoda ... tak kazdy step se udela render zde..
            action = pi[state]
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed)

    main(env, args)
