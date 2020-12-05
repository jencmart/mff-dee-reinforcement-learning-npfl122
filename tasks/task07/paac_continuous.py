#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import shutil
import zipfile
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.9999, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0009, type=float, help="Learning rate.")
parser.add_argument("--tiles", default=4, type=int, help="Tiles to use.")
parser.add_argument("--workers", default=16, type=int, help="Number of parallel workers.")
parser.add_argument("--embedding", default=64, type=int, help="Number of parallel workers.")

# 8194b193-e909-11e9-9ce9-00505601122b
# 47b0acaf-eb3e-11e9-9ce9-00505601122b


@tf.function
def custom_loss(y_true, y_pred, returns=0, predictions=0):
    # TODO:
    #  Run the model on given `states` and compute sds, mus and predicted values.
    #  Then create `action_distribution` using `tfp.distributions.Normal` class and computed mus and sds.

    # TODO: Compute total loss as a sum of three losses:
    #  - negative log likelihood of the `actions` in the `action_distribution` (using the `log_prob` method).
    #    You then sum the log probs of actions in a single batch example (`tf.math.reduce_sum` with `axis=1`).
    #    Finally multiply the resulting vector by (returns - predicted values) and compute its mean.
    #    Note that the gradient must NOT flow through the predicted values (use `tf.stop_gradient` if necessary).
    #  - negative value of the distribution entropy (use `entropy` method of
    #    the `action_distribution`) weighted by `args.entropy_regularization`.
    #  - mean square error of the `returns` and predicted values.

    action_dist = tfp.distributions.Normal(y_true[0], y_true[1])
    entropy = action_dist.entropy()
    regularization = args.entropy_regularization * entropy
    log_probs = action_dist.log_prob(y_pred)
    log_probs_weighted = log_probs * (returns - predictions)
    loss = - log_probs_weighted - regularization
    return loss


class Network:
    def __init__(self, env, args):
        # TODO: Analogously to paac, your model should contain two components:
        #  - actor, which predicts distribution over the actions
        #  - critic, which predicts the value function
        self.n_classes = env.observation_space.nvec[-1]
        # TODO:
        #  The given states are tile encoded, so they are indices of
        #  tiles intersecting the state. Therefore, you should convert them
        #  to dense encoding (one-hot-like, with with `args.tiles` ones).
        #  (Or you can even use embeddings for better efficiency.)

        # TODO:
        #  The actor computes `mus` and `sds`, each of shape [batch_size, actions].
        #  Compute each independently using states as input, adding a fully connected
        #  layer with `args.hidden_layer_size` units and ReLU activation. Then:
        #  - For `mus`, add a fully connected layer with `actions` outputs.
        #    To avoid `mus` moving from the required range, you should apply
        #    properly scaled `tf.tanh` activation.
        #  - For `sds`, add a fully connected layer with `actions` outputs
        #    and `tf.nn.softplus` activation.

        # the state itself is a list of indices of tiles ... there are always args.tiles=8 nonzero indices
        # not sure but now the input is a matrix of 0 and 1
        # one column of it has args.tiles zeros or ones
        # its ok one column is one feature (feature is a combination of tiles ... some tile encoding)
        # so if there is 1 on the position i,j in the matrix this because in the original list of indices
        # on the position i was index j
        input_layer = tf.keras.layers.Input(env.observation_space.shape[0], env.observation_space.nvec[-1])
        embedding = tf.keras.layers.Embedding(input_dim=env.observation_space.nvec[-1],
                                              output_dim=args.embedding,
                                              input_length=env.observation_space.shape[0])(input_layer)
        hidden_layer = tf.keras.layers.Flatten()(embedding)
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(hidden_layer)
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size // 8, activation=tf.nn.relu)(hidden_layer)
        output_mu = tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.nn.tanh)(hidden_layer)
        output_sd = tf.keras.layers.Dense(env.action_space.shape[0])(hidden_layer)
        output_sd = tf.math.softplus(output_sd)
        output_sd = tf.add(output_sd, tf.constant(1e-9))
        self.actor = tf.keras.Model(input_layer, [output_mu, output_sd])
        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            run_eagerly=False
        )

        # TODO:
        #  The critic should be a usual one, passing states through one hidden
        #  layer with `args.hidden_layer_size` ReLU units and then predicting
        #  the value function.
        input_layer = tf.keras.layers.Input(env.observation_space.shape[0], env.observation_space.nvec[-1])
        embedding = tf.keras.layers.Embedding(input_dim=env.observation_space.nvec[-1],
                                              output_dim=args.embedding,
                                              input_length=env.observation_space.shape[0])(input_layer)
        hidden_layer = tf.keras.layers.Flatten()(embedding)
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(hidden_layer)
        hidden_layer = tf.keras.layers.Dense(args.hidden_layer_size // 8, activation=tf.nn.relu)(hidden_layer)

        output_layer = tf.keras.layers.Dense(1)(hidden_layer)
        self.critic = tf.keras.Model(input_layer, output_layer)
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            run_eagerly=False
        )

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train(self, states, actions, returns):
        actions = tf.convert_to_tensor(actions)
        self.actor.optimizer.minimize(
            lambda: custom_loss(self.actor(states, training=False), actions, returns, self.critic(states)),
            var_list=self.actor.trainable_variables
        )
        self.critic.optimizer.minimize(
            lambda: self.critic.loss(self.critic(states, training=True), returns),
            var_list=self.critic.trainable_variables
        )

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states):
        # TODO: Return predicted action distributions (mus and sds).
        result = self.actor(states)
        return result

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        # TODO: Return predicted state-action values.
        result = self.critic(states)
        return result


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

    def evaluate_episode(start_evaluation=False):
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Predict the action using the greedy policy
            # can just take mu
            # actions = network.predict_actions(np.asarray([state]))[0]
            # state = np.asarray([get_one_hot(np.array(state), env.observation_space.nvec[-1])])
            action, _ = network.predict_actions(np.asarray([state]))
            # action, _ = network.predict_actions(state)
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.vector.AsyncVectorEnv(
        [lambda: wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"),
                                                     tiles=args.tiles)] * args.workers)
    vector_env.seed(args.seed)
    # states is a matrix of n_workers row and n_tiles columns
    # so each row corresponds to some environment
    states = vector_env.reset()
    best_so_far = 90
    all_returns = []
    training = not args.recodex
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Predict action distribution using `network.predict_actions`
            #  and then sample it using for example `np.random.normal`. Do not
            #  forget to clip the actions to the `env.action_space.{low,high}`
            #  range, for example using `np.clip`.

            # no need to select between greedy and non greedy actions ... perform sampling
            mus, sds = network.predict_actions(states)
            actions = [np.random.normal(mu, sd) for mu, sd in zip(mus, sds)]
            actions = np.clip(actions, env.action_space.low[0], env.action_space.high[0])

            # TODO(paac): Perform steps in the vectorized environment
            new_states, rewards, dones, _ = vector_env.step(actions)

            # TODO(paac): Compute estimates of returns by one-step bootstrapping
            predictions = network.predict_values(new_states)
            predictions = args.gamma * predictions.flatten()
            episode_returns = rewards + (1 - dones) * predictions

            # TODO(paac): Train network using current states, chosen actions and estimated returns
            episode_returns = np.expand_dims(episode_returns, 1)
            network.train(states, actions, episode_returns)
            states = new_states

        # Periodic evaluation
        for _ in range(args.evaluate_for):
            all_returns.append(evaluate_episode())

        if len(all_returns) > 200 and np.average(all_returns[-150:]) > best_so_far:
            best_so_far = np.average(all_returns[-150:])
            print("Best so far: {:.5}".format(best_so_far))
            save_model(network.actor, "paac_cont_actor_model")
            save_model(network.critic, "paac_cont_critic_model")

    network.actor = load_model('paac_cont_actor_model')
    network.critic = load_model('paac_cont_critic_model')

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(
        wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=args.tiles), args.seed)

    main(env, args)
