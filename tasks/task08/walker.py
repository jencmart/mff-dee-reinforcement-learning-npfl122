#!/usr/bin/env python3
import argparse
import collections
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")  # Report only TF errors by default

import os

import gym
import numpy as np
import tensorflow as tf
import shutil
import wrappers
import zipfile

# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=1, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")  # try 64
parser.add_argument("--evaluate_each", default=500, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")

parser.add_argument("--polyak", default=0.005, type=float, help="Polyak...")

parser.add_argument("--explore_noise", default=0.1, type=float, help="Polyak...")

parser.add_argument("--target_delay", default=2, type=int, help="delay target policy and target Q")
parser.add_argument("--policy_noise_sd", default=0.2, type=float, help="noise to policy")
parser.add_argument("--policy_noise_clip", default=0.5, type=float, help="clip high policy noise")


class Network:

    def build_compile_Q_critic(self, state_dim, action_dim, learning_rate, i):

        # input for action ... [batch, a] i.e. [64, 4]
        input_actions = tf.keras.layers.Input(action_dim, name='critic_q_{}_input_actions'.format(i))
        # input for state ... [batch, s] i.e. [64, 24]
        input_states = tf.keras.layers.Input(state_dim, name='critic_q_{}_input_states'.format(i))

        # Common hidden layers ...
        hidden_layer = tf.keras.layers.Concatenate(name='critic_q_{}_concatenation'.format(i))([input_states, input_actions])
        hidden_layer = tf.keras.layers.Dense(400, activation=tf.nn.relu, name='critic_q_{}_hidden1_common'.format(i))(hidden_layer)
        hidden_layer = tf.keras.layers.Dense(300, activation=tf.nn.relu, name='critic_q_{}_hidden2_common'.format(i))(hidden_layer)
        output_layer = tf.keras.layers.Dense(1, name='critic_q_{}_output'.format(i))(hidden_layer)  #  1 q value for a \in R^n

        #                           [batch, 24]   [batch, 4]
        q = tf.keras.models.Model([input_states, input_actions], output_layer, name="CRITIC_Q_{}_MODEL".format(i))

        q.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        return q

    def build_compile_actor(self, args, state_dim, action_dim, max_action):
        input_layer = tf.keras.layers.Input(state_dim)  # ... 24 ...
        hidden_layer = tf.keras.layers.Dense(400, activation=tf.nn.relu)(input_layer)
        hidden_layer = tf.keras.layers.Dense(300, activation=tf.nn.relu)(hidden_layer)
        output_layer = tf.keras.layers.Dense(action_dim, activation=tf.nn.tanh)(hidden_layer)
        output_layer = tf.multiply(output_layer, max_action)

        policy_actor = tf.keras.Model(input_layer, output_layer)

        policy_actor.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), )
        return policy_actor

    def compile(self):
        self.policy_actor.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), )

        self.q1_critic.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        self.q2_critic.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=tf.keras.losses.MeanSquaredError(),
        )


    def __init__(self, env, args, state_dim, action_dim, max_action):
        self.polyak = args.polyak
        self.env = env
        self.lr = args.learning_rate
        self.policy_actor = self.build_compile_actor(args, state_dim, action_dim, max_action)
        self.target_policy_actor = tf.keras.models.clone_model(self.policy_actor)

        self.q1_critic = self.build_compile_Q_critic(state_dim, action_dim, args.learning_rate, 1)
        self.target_q1_critic = tf.keras.models.clone_model(self.q1_critic)

        self.q2_critic = self.build_compile_Q_critic(state_dim, action_dim, args.learning_rate, 2)
        self.target_q2_critic = tf.keras.models.clone_model(self.q2_critic)

        self.max_action = 1
        self.policy_noise_sd = args.policy_noise_sd
        self.policy_noise_clip = args.policy_noise_clip

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states):
        return self.policy_actor(states)

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train_critic_Q(self, states, actions, returns):
        with tf.GradientTape() as tape:
            # predict
            critic1_values = self.q1_critic([states, actions], training=True)
            loss1 = tf.reduce_mean(tf.math.square(returns - critic1_values))
        q1_grad = tape.gradient(loss1, self.q1_critic.trainable_variables)
        self.q1_critic.optimizer.apply_gradients(zip(q1_grad, self.q1_critic.trainable_variables))

        with tf.GradientTape() as tape:
            critic2_values = self.q2_critic([states, actions], training=True)
            loss2 = tf.reduce_mean(tf.math.square(returns - critic2_values))
        q2_grad = tape.gradient(loss2, self.q2_critic.trainable_variables)
        self.q2_critic.optimizer.apply_gradients(zip(q2_grad, self.q2_critic.trainable_variables))

        return loss1, loss2

        # self.q1_critic.optimizer.minimize(
        #     lambda: self.q1_critic.loss(returns, self.q1_critic([states, actions], training=True)),  # q_old, q_new
        #     var_list=self.q1_critic.trainable_variables
        # )
        #
        # # TODO TD3 - train Q2
        # self.q2_critic.optimizer.minimize(
        #     lambda: self.q2_critic.loss(returns, self.q2_critic([states, actions], training=True)),  # q_old, q_new
        #     var_list=self.q2_critic.trainable_variables
        # )

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def train_actor(self, states):
        # TODO TD3 - actor updated only using q1
        with tf.GradientTape() as tape:
            actions = self.policy_actor(states, training=True)
            critic1_value = self.q1_critic([states, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic1_value)

        actor_grad = tape.gradient(actor_loss, self.policy_actor.trainable_variables)
        self.policy_actor.optimizer.apply_gradients(zip(actor_grad, self.policy_actor.trainable_variables))
        return actor_loss


    @tf.function
    def polyak_target_update(self):
        for var, target_var in zip(self.policy_actor.trainable_variables, self.target_policy_actor.trainable_variables):
            target_var.assign(target_var * (1 - self.polyak) + var * self.polyak)

        for var, target_var in zip(self.q1_critic.trainable_variables, self.target_q1_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.polyak) + var * self.polyak)

        # TODO TD3 - target Q2 update ...
        for var, target_var in zip(self.q2_critic.trainable_variables, self.target_q2_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.polyak) + var * self.polyak)


    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        predicted_actions = self.target_policy_actor(tf.convert_to_tensor(states))

        # TODO TD3: smoothing of target policy
        noise = tf.random.normal(shape=tf.shape(predicted_actions), mean=0.0, stddev=self.policy_noise_sd, dtype=tf.float32)
        noise = tf.clip_by_value(noise, -1*self.policy_noise_clip, self.policy_noise_clip)
        predicted_actions = predicted_actions + noise
        predicted_actions = tf.clip_by_value(predicted_actions, -1*self.max_action, self.max_action)

        # TODO TD3: take smaller from target_q1 target_q2
        #                             Bx24            Bx1
        #                            [batch, 24]   [batch,1]
        q1 = self.target_q1_critic([states, predicted_actions])  # model.predict([testAttrX, testImagesX])
        q2 = self.target_q2_critic([states, predicted_actions])
        target_Q = tf.math.minimum(q1, q2)
        return target_Q

    def train(self, replay_buffer, K):
        for kk in range(K):
            batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
            states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))

            # calculate Q
            target_Q = self.predict_values(np.asarray(next_states))
            returns = rewards + ((1 - dones) * args.gamma * target_Q)

            # train ciric 1 and critic 2
            q1_loss, q2_loss = self.train_critic_Q(states, actions, returns)

            # TODO TD3: Delayed Updates
            if kk % args.target_delay == 0:

                # update critic
                actor_loss = self.train_actor(states)

                # update polyak target_q1, target_q2, target_critic
                self.polyak_target_update()
        return actor_loss, q1_loss, q2_loss

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


def load_network(network):
    # loading the models
    network.policy_actor = load_model("actor_model")
    network.q1_critic = load_model("critic_model")
    network.q2_critic = load_model("critic_2_model")

    network.target_policy_actor = load_model("target_actor_model")
    network.target_q1_critic = load_model("target_critic_model")
    network.target_q2_critic = load_model("target_critic_2_model")


def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1
    network = Network(env, args, state_dim, action_dim, max_action)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque(maxlen=10000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def evaluate_episode(start_evaluation=False):
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Predict the action using the greedy policy
            action = network.predict_actions(np.asarray([state]))[0]
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    training = not args.recodex
    all_returns = []
    best_so_far = -30
    try:
        load_network(network)
        network.compile()
    except:
        pass
    ep = 0
    while training:
        # Training
        for E in range(args.evaluate_each):
            state, done = env.reset(), False
            er_ret = 0
            ep += 1
            MAX_ = 2000
            K=0
            actor_losses = []
            q1_losses = []
            q2_losses = []
            for K in range(MAX_):
                if E % 50 == 0:
                    env.render()
                # Select and add noise to action
                single_state_batch = np.asarray([state])
                action = network.predict_actions(single_state_batch)[0]
                action = action + np.random.normal(0, args.explore_noise, size=4)
                action = np.clip(action, -1*max_action, max_action)

                next_state, reward, done, _ = env.step(action)
                if reward > -100:
                    er_ret += reward
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if (done or K+1 == MAX_) and len(replay_buffer) >= args.batch_size:
                    actor_loss, q1_loss, q2_loss = network.train(replay_buffer, K)
                    actor_losses.append(actor_loss)
                    q1_losses.append(q1_loss)
                    q2_losses.append(q2_loss)
                if done or K+1 == MAX_:
                    break
            if len(actor_losses) > 0:
                print('\rEpisode: {},\tDist.: {:.2f},\tactor_loss: {:.10f},\tc1_loss:{:.10f},\tc2_loss:{:.10f}' .format(ep, er_ret, np.mean(actor_losses), np.mean(q1_losses),np.mean(q2_losses), end=""))
        print("Periodic Evaluation ........................................")
        # Periodic evaluation
        for _ in range(args.evaluate_for):
            r = evaluate_episode()
            all_returns.append(r)

        avg = sum(all_returns[-20:])/20
        print("........ Current mean {}-episode return: {}".format(args.evaluate_for, avg))
        if avg > best_so_far:
            best_so_far = avg
            print("Best so far: {}".format(best_so_far))

            save_model(network.policy_actor, "actor_model")
            save_model(network.q1_critic, "critic_model")
            save_model(network.q2_critic, "critic_2_model")

            save_model(network.target_policy_actor, "target_actor_model")
            save_model(network.target_q1_critic, "target_critic_model")
            save_model(network.target_q2_critic, "target_critic_2_model")

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("BipedalWalker-v3"), args.seed)
    main(env, args)
