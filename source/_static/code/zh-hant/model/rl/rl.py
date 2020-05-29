import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque

num_episodes = 500              # 遊戲訓練的總episode數量
num_exploration_episodes = 100  # 探索過程所占的episode數量
max_len_episode = 1000          # 每個episode的最大回合數
batch_size = 32                 # 批次大小
learning_rate = 1e-3            # 學習率
gamma = 1.                      # 折扣因子
initial_epsilon = 1.            # 探索起始時的探索率
final_epsilon = 0.01            # 探索終止時的探索率

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


if __name__ == '__main__':
    env = gym.make('CartPole-v1')       # 實例化一個遊戲環境，參數為遊戲名稱
    model = QNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = deque(maxlen=10000) # 使用一個 deque 作為 Q Learning 的經驗回放池
    epsilon = initial_epsilon
    for episode_id in range(num_episodes):
        state = env.reset()             # 初始化環境，獲得初始狀態
        epsilon = max(                  # 計算當前探索率
            initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
            final_epsilon)
        for t in range(max_len_episode):
            env.render()                                # 對當前影像進行渲染，繪圖到影像
            if random.random() < epsilon:               # epsilon-greedy 探索策略，以 epsilon 的機率選擇隨機動作
                action = env.action_space.sample()      # 選擇隨機動作（探索）
            else:
                action = model.predict(np.expand_dims(state, axis=0)).numpy()   # 選擇模型計算出的 Q Value 最大的動作
                action = action[0]

            # 讓環境執行動作，獲得執行完動作的下一個狀態，動作的獎勵，遊戲是否已結束以及額外資訊
            next_state, reward, done, info = env.step(action)
            # 如果遊戲Game Over，給予大的負獎勵
            reward = -10. if done else reward
            # 將(state, action, reward, next_state)的四元組（外加 done 標籤表示是否結束）放入經驗回放池
            replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
            # 更新當前 state
            state = next_state

            if done:                                    # 遊戲結束則退出本輪循環，進行下一個 episode
                print("episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
                break

            if len(replay_buffer) >= batch_size:
                # 從經驗回放池中隨機取一個批次的四元組，並分別轉換為 NumPy 陣列
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
                    *random.sample(replay_buffer, batch_size))
                batch_state, batch_reward, batch_next_state, batch_done = \
                    [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
                batch_action = np.array(batch_action, dtype=np.int32)

                q_value = model(batch_next_state)
                y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)  # 計算 y 值
                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mean_squared_error(  # 最小化 y 和 Q-value 的距離
                        y_true=y,
                        y_pred=tf.reduce_sum(model(batch_state) * tf.one_hot(batch_action, depth=2), axis=1)
                    )
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))       # 計算梯度並更新參數