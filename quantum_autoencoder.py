%env TF_FORCE_GPU_ALLOW_GROWTH=true
%env WANDB_AGENT_MAX_INITIAL_FAILURES=1024

import wandb
import gymnasium as gym
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

tf.config.optimizer.set_jit(True)  # Enable XLA.

wandb.login()

sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "epochs": {"value": 2000},
        "buffer_size": {"value": 1000000},
        "batch_size": {"value": 256},
        "lr": {"value": 3e-4},
        "global_clipnorm": {"value": 1.0},
        "tau": {"value": 0.01},
        "gamma": {"value": 0.99},
        "temp_init": {"value": 1.0},
        "temp_min": {"value": 0.01},
        "temp_decay": {"value": 1e-5},
    },
}

sweep_id = wandb.sweep(sweep_config, project="Quantum_Autoencoder")

class DuelingDQN(tf.keras.Model):
    def __init__(self, action_space):
        super(DuelingDQN, self).__init__()

        self.fc1 = tf.keras.layers.Dense(
            512,
            activation="elu",
            kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
        )
        self.fc2 = tf.keras.layers.Dense(
            256,
            activation="elu",
            kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
        )
        self.V = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
        )
        self.A = tf.keras.layers.Dense(
            action_space,
            activation=None,
            kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
        )

    def call(self, inputs, training=None):
        x = self.fc1(inputs, training=training)
        x = self.fc2(x, training=training)
        V = self.V(x, training=training)
        A = self.A(x, training=training)
        adv_mean = tf.reduce_mean(A, axis=-1, keepdims=True)
        return V + (A - adv_mean)

    def get_action(self, state, temperature):
        return tf.random.categorical(self(state) / temperature, 1)[0, 0]

class IntrinsicModel(tf.keras.Model):
    def __init__(self):
        super(IntrinsicModel, self).__init__()

        # init reward normalization
        self.rew_rms = tf.keras.layers.Normalization()

    def build(self, input_shape):
        # Encoder
        self._encoder = [
            tf.keras.layers.Dense(
                128,
                activation="elu",
                kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
            ),
            tf.keras.layers.Dense(
                64,
                activation="elu",
                kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
            ),
            tf.keras.layers.Dense(
                32,
                activation="elu",
                kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
            ),
            tf.keras.layers.Dense(
                16,
                activation="elu",
                kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
            ),
            # Latent space
            tf.keras.layers.Dense(
                (input_shape[-1] // 2),
                activation=None,
                kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
                name="latent_space",
            ),
        ]
        # Decoder
        self._decoder = [
            tf.keras.layers.Dense(
                16,
                activation="elu",
                kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
            ),
            tf.keras.layers.Dense(
                32,
                activation="elu",
                kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
            ),
            tf.keras.layers.Dense(
                64,
                activation="elu",
                kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
            ),
            tf.keras.layers.Dense(
                128,
                activation="elu",
                kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
            ),
            tf.keras.layers.Dense(
                input_shape[-1],
                activation=None,
                kernel_initializer=tf.keras.initializers.Orthogonal(tf.sqrt(2.0)),
                name="reconstruction",
            ),
        ]
        self.rew_rms.build((None, 1))
        super(IntrinsicModel, self).build(input_shape)

    def call(self, inputs, training=None, only_encoder=None):
        for l in self._encoder:
            inputs = l(inputs, training=training)
        if not only_encoder:
            for l in self._decoder:
                inputs = l(inputs, training=training)
        return inputs

    def get_int_reward(self, inputs):
        inputs = tf.cast(inputs, dtype=self.dtype)
        y_pred = self(inputs, training=False)
        reward = tf.reduce_sum(tf.square(inputs - y_pred), axis=-1, keepdims=True)

        # update reward statistics
        self.rew_rms.update_state(reward)
        self.rew_rms.finalize_state()

        # normalize intrinsic reward
        reward = tf.nn.relu6(self.rew_rms(reward))

        return reward

class ReplayBuffer:
    def __init__(self, shape, size=1e6):
        self.size = int(size)
        self.counter = 0
        self.state_buffer = np.zeros((self.size, *shape), dtype=np.float32)
        self.action_buffer = np.zeros(self.size, dtype=np.int32)
        self.int_reward_buffer = np.zeros(self.size, dtype=np.float32)
        self.ext_reward_buffer = np.zeros(self.size, dtype=np.float32)
        self.new_state_buffer = np.zeros((self.size, *shape), dtype=np.float32)
        self.terminal_buffer = np.zeros(self.size, dtype=np.bool_)

    def __len__(self):
        return self.counter

    def add(self, state, action, ext_reward, int_reward, new_state, done):
        idx = self.counter % self.size
        self.state_buffer[idx] = state
        self.action_buffer[idx] = action
        self.int_reward_buffer[idx] = int_reward
        self.ext_reward_buffer[idx] = ext_reward
        self.new_state_buffer[idx] = new_state
        self.terminal_buffer[idx] = done
        self.counter += 1

    def sample(self, batch_size):
        max_buffer = min(self.counter, self.size)
        batch = np.random.choice(max_buffer, batch_size, replace=False)
        state_batch = self.state_buffer[batch]
        action_batch = self.action_buffer[batch]
        int_reward_batch = self.int_reward_buffer[batch]
        ext_reward_batch = self.ext_reward_buffer[batch]
        new_state_batch = self.new_state_buffer[batch]
        done_batch = self.terminal_buffer[batch]

        return (
            state_batch,
            action_batch,
            ext_reward_batch,
            int_reward_batch,
            new_state_batch,
            done_batch,
        )

def update_target(net, net_targ, tau):
    for source_weight, target_weight in zip(
        net.trainable_variables, net_targ.trainable_variables
    ):
        target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

def train_step(dqn, target_dqn, int_model, replay_buffer, batch_size, tau, gamma):
    (
        states,
        actions,
        ext_rewards,
        int_rewards,
        next_states,
        dones,
    ) = replay_buffer.sample(batch_size)

    # predict next Q
    next_Q = target_dqn(next_states)
    next_Q = tf.reduce_max(next_Q, axis=-1)

    # get targets
    targets = np.array(dqn(states))
    # for experiments with only extrinsic reward, use 'ext_rewards' instead of 'int_rewards'
    targets[np.arange(batch_size), actions] = int_rewards + (
        (1.0 - tf.cast(dones, dtype=tf.float32)) * gamma * next_Q
    )

    # update dqn
    dqn_loss = dqn.train_on_batch(states, targets)

    # update int model
    int_loss = int_model.train_on_batch(states, states)

    # soft update target Q
    update_target(dqn, target_dqn, tau=tau)

    return dqn_loss, int_loss

def run(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # make an environment
        # env = gym.make("CartPole-v1")
        env = gym.make('MountainCar-v0')
        # env = gym.make("LunarLander-v2")
        # env = gym.make('Acrobot-v1')

        state_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        # init variables
        total_steps = 0
        temp = config.temp_init

        # init models
        q_model = DuelingDQN(action_space)
        q_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.lr, global_clipnorm=config.global_clipnorm
            ),
            loss="mean_squared_error",
        )
        q_targ_model = DuelingDQN(action_space)
        q_targ_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.lr, global_clipnorm=config.global_clipnorm
            ),
            loss="mean_squared_error",
        )
        update_target(q_model, q_targ_model, tau=1.0)

        i_model = IntrinsicModel()
        i_model.build((None, state_space))
        i_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.lr, global_clipnorm=config.global_clipnorm
            ),
            loss="mean_squared_error",
        )

        # init replay buffer
        r_buffer = ReplayBuffer(shape=(state_space,), size=config.buffer_size)

        # train
        for e in range(config.epochs):
            state = env.reset()
            done = False

            ep_ext_reward = 0
            ep_int_reward = 0

            while not done:
                action = q_model.get_action(state, temp)
                new_state, ext_reward, done, _ = env.step(action)

                int_reward = i_model.get_int_reward(state.reshape(1, -1))

                r_buffer.add(
                    state,
                    action,
                    ext_reward,
                    int_reward.numpy()[0][0],
                    new_state,
                    done,
                )

                if len(r_buffer) > config.batch_size:
                    dqn_loss, int_loss = train_step(
                        q_model,
                        q_targ_model,
                        i_model,
                        r_buffer,
                        config.batch_size,
                        config.tau,
                        config.gamma,
                    )
                    wandb.log(
                        {"DQN loss": dqn_loss, "Intrinsic loss": int_loss},
                        step=total_steps,
                    )

                ep_ext_reward += ext_reward
                ep_int_reward += int_reward.numpy()[0][0]

                state = new_state
                total_steps += 1

                temp = max(
                    config.temp_min, temp * np.exp(-config.temp_decay * total_steps)
                )

            wandb.log(
                {
                    "Extrinsic reward": ep_ext_reward,
                    "Intrinsic reward": ep_int_reward,
                    "Temperature": temp,
                },
                step=e,
            )

        env.close()

wandb.agent(sweep_id, function=run, count=1)
