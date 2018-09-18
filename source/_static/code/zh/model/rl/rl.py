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


# Q-network用于拟合Q函数，和前节的多层感知机类似。输入state，输出各个action下的Q-value（CartPole下为2维）。
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


env = gym.make('CartPole-v1')       # 实例化一个游戏环境，参数为游戏名称
model = QNetwork()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
replay_buffer = deque(maxlen=10000)
epsilon = initial_epsilon
for episode_id in range(num_episodes):
    state = env.reset()             # 初始化环境，获得初始状态
    epsilon = max(
        initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
        final_epsilon)
    for t in range(max_len_episode):
        env.render()                # 对当前帧进行渲染，绘图到屏幕
        if random.random() < epsilon:               # epsilon-greedy探索策略
            action = env.action_space.sample()      # 以epsilon的概率选择随机动作
        else:
            action = model.predict(
                tf.constant(np.expand_dims(state, axis=0), dtype=tf.float32)).numpy()
            action = action[0]
        next_state, reward, done, info = env.step(action)               # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
        reward = -10. if done else reward                               # 如果游戏Game Over，给予大的负奖励
        replay_buffer.append((state, action, reward, next_state, 1 if done else 0)) # 将(state, action, reward, next_state)的四元组（外加done标签表示是否结束）放入经验重放池
        state = next_state

        if done:                                                        # 游戏结束则退出本轮循环，进行下一个episode
            print("episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
            break

        if len(replay_buffer) >= batch_size:
            # 从经验回放池中随机取一个batch的四元组，并分别转换为NumPy数组
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
                *random.sample(replay_buffer, batch_size))
            batch_state, batch_reward, batch_next_state, batch_done = \
                [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
            batch_action = np.array(batch_action, dtype=np.int32)

            q_value = model(tf.constant(batch_next_state, dtype=tf.float32))
            y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)  # 按照论文计算y值
            with tf.GradientTape() as tape:
                loss = tf.losses.mean_squared_error(        # 最小化y和Q-value的距离
                    labels=y,
                    predictions=tf.reduce_sum(model(tf.constant(batch_state)) *
                                              tf.one_hot(batch_action, depth=2), axis=1)
                )
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))       # 计算梯度并更新参数