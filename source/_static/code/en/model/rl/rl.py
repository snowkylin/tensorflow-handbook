import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque

tf.enable_eager_execution()
num_episodes = 500
num_exploration_episodes = 100
max_len_episode = 1000
batch_size = 32
learning_rate = 1e-3
gamma = 1.
initial_epsilon = 1.
final_epsilon = 0.01


# Q-network is used to fit Q function resemebled as the aforementioned multilayer perceptron. It inputs state and output Q-value under each action (2 dimensional under CartPole).
class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)


env = gym.make('CartPole-v1')       # Instantiate a game environment. The parameter is its name.
model = QNetwork()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
replay_buffer = deque(maxlen=10000)
epsilon = initial_epsilon
for episode_id in range(num_episodes):
    state = env.reset()             # Initialize the environment and get its initial state.
    epsilon = max(
        initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
        final_epsilon)
    for t in range(max_len_episode):
        env.render()                # Render the current frame.
        if random.random() < epsilon:               # Epsilon-greedy exploration strategy.
            action = env.action_space.sample()      # Choose random action with the probability of epilson.
        else:
            action = model.predict(
                tf.constant(np.expand_dims(state, axis=0), dtype=tf.float32)).numpy()
            action = action[0]
        next_state, reward, done, info = env.step(action)               # Let the environment to execute the action, get the next state of the action, the reward of the action, whether the game is done and extra information.
        reward = -10. if done else reward                               # Give a large negative reward if the game is over.
        replay_buffer.append((state, action, reward, next_state, 1 if done else 0)) # Put the (state, action, reward, next_state) quad back into the experience replay pool.
        state = next_state

        if done:                                                        # Exit this round and enter the next episode if the game is over.
            print("episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
            break

        if len(replay_buffer) >= batch_size:
            # Randomly take a batch quad from the experience replay pool and transform them to NumPy array respectively.
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
                *random.sample(replay_buffer, batch_size))
            batch_state, batch_reward, batch_next_state, batch_done = \
                [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
            batch_action = np.array(batch_action, dtype=np.int32)

            q_value = model(tf.constant(batch_next_state, dtype=tf.float32))
            y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)  # Calculate y according to the method in the paper.
            with tf.GradientTape() as tape:
                loss = tf.losses.mean_squared_error(        # Minimize the distance between y and Q-value.
                    labels=y,
                    predictions=tf.reduce_sum(model(tf.constant(batch_state)) *
                                              tf.one_hot(batch_action, depth=2), axis=1)
                )
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))       # Calculate the gradient and update parameters.