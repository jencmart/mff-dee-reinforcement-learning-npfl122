#!/usr/bin/env python3
import argparse
import collections
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Report only TF errors by default
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
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")  # try 64
parser.add_argument("--env", default="Pendulum-v0", type=str, help="Environment.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0008, type=float, help="Learning rate.")
parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")


class Network:
    def __init__(self, env, args):
        # TODO: Create:
        #  - an actor, which starts with states and returns actions.
        #    Usually, one or two hidden layers are employed. As in the
        #    paac_continuous, to keep the actions in the required range, you
        #    should apply properly scaled `tf.tanh` activation.
        self.actor = None
        input_layer = tf.keras.layers.Input(env.observation_space.shape[0])
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(input_layer)
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size // 2, activation=tf.nn.relu)(hidden_layer)
        output_layer = tf.keras.layers.Dense(1, activation=tf.nn.tanh)(hidden_layer)
        output_layer = tf.multiply(output_layer, 2)
        self.actor = tf.keras.Model(input_layer, output_layer)
        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            # run_eagerly=True
        )
        # TODO: Create:
        #  - a target actor as the copy of the actor using `tf.keras.models.clone_model`.
        self.target_actor = tf.keras.models.clone_model(self.actor)

        # TODO: Create:
        #  - a critic, starting with given states and actions producing predicted
        #   returns.  Usually, states are fed through a hidden layer first, and
        #   then concatenated with action and fed through two more hidden
        #   layers, before computing the returns.

        self.critic = None
        input_actions = tf.keras.layers.Input(env.action_space.shape[0])
        hidden_layer_actions = tf.keras.layers.Dense(args.hidden_layer_size // 4, activation=tf.nn.relu, name='hidden1_action')(input_actions)
        input_states = tf.keras.layers.Input(env.observation_space.shape[0])
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size // 4, activation=tf.nn.relu, name='hidden1_state')(input_states)
        hidden_layer = tf.keras.layers.Concatenate(name='concatenation')([hidden_layer, hidden_layer_actions])
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu, name='hidden1_common')(hidden_layer)
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size // 2, activation=tf.nn.relu, name='hidden2_common')(hidden_layer)
        output_layer = tf.keras.layers.Dense(1, name='output')(hidden_layer)
        self.critic = tf.keras.models.Model([input_states, input_actions], output_layer)
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            # run_eagerly=True
        )

        # TODO: Create:
        #  - a target critic as the copy of the critic using `tf.keras.models.clone_model`.
        self.target_critic = tf.keras.models.clone_model(self.critic)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states):
        # TODO: Return predicted actions by the actor.
        if states.shape != (1,3):
            print(states.shape)
        return self.actor(states)

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train(self, states, actions, returns):
        # TODO: Separately train:
        #  - the critic using MSE loss.
        #  - the actor using the DPG loss,

        # try train_on_batch(x,y)
        # targets are given by the target network ... namely target_critic if we are training critic network
        # self.critic.train_on_batch([states, actions], returns)
        self.critic.optimizer.minimize(
            lambda: self.critic.loss(returns, self.critic([states, actions], training=True)),  # q_old, q_new
            var_list=self.critic.trainable_variables
        )
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            critic_value = self.critic([states, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )
        # TODO:
        #  Furthermore, update the target actor and critic networks by
        #  exponential moving average with weight `args.target_tau`. A possible
        #  way to implement it inside a `tf.function` is the following:
        #    for var, target_var in zip(network.trainable_variables, target_network.trainable_variables):
        #        target_var.assign(target_var * (1 - target_tau) + var * target_tau)
        self.target_update()

    @tf.function
    def target_update(self):
        for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)

        for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        # TODO: Return predicted returns -- predict actions by the target actor
        #  and evaluate them using the target critic.
        predicted_actions = self.target_actor(tf.convert_to_tensor([states]))
        # print('done')
        predicted_returns = self.target_critic([tf.convert_to_tensor([states]), predicted_actions])
        return predicted_returns


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


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


def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
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

    noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
    training = not args.recodex
    all_returns = []
    best_so_far = -1400
    while training:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            noise.reset()
            while not done:
                # TODO: Predict actions by calling `network.predict_actions`
                #  and adding the Ornstein-Uhlenbeck noise. As in paac_continuous,
                #  clip the actions to the `env.action_space.{low,high}` range.
                action = network.predict_actions(np.asarray([state]))[0] + noise.sample()
                # check: env.action_space.low[0], env.action_space.high[0]
                action = np.clip(action, env.action_space.low[0], env.action_space.high[0])

                next_state, reward, done, _ = env.step(action)
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) >= args.batch_size:
                    batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                    states, actions, rewards, dones, next_states = map(np.array,
                                                                       zip(*[replay_buffer[i] for i in batch]))
                    # TODO: Perform the training
                    returns = rewards + args.gamma * network.predict_values(np.asarray([next_states]))
                    network.train(states, actions, returns)

        # Periodic evaluation
        for _ in range(args.evaluate_for):
            all_returns.append(evaluate_episode())

        if sum(all_returns[-20:])/20 > best_so_far:
            best_so_far = sum(all_returns[-20:])/20
            print("Best so far: {}".format(best_so_far))

            save_model(network.actor, "actor_model")
            save_model(network.critic, "critic_model")
            save_model(network.target_actor, "target_actor_model")
            save_model(network.target_critic, "target_critic_model")

    # loading the models
    network.actor = load_model("actor_model")
    network.critic = load_model("critic_model")
    network.target_actor = load_model("target_actor_model")
    network.target_critic = load_model("target_critic_model")

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make(args.env), args.seed)

    main(env, args)
