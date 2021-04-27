import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym

num_episodes = 500
max_len_episode = 1000
learning_rate = 1e-3
gamma = 1.
num_actions = 2

policy_network = tf.keras.Sequential([
    tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=num_actions, activation=tf.nn.softmax),
])

baseline_network = tf.keras.Sequential([
    tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=1),
])

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for episode_id in range(num_episodes):
        state = env.reset()
        trajectory = []     # 记录轨迹
        for t in range(max_len_episode):
            env.render()
            probs = policy_network(np.expand_dims(state, axis=0))[0]
            action = tfp.distributions.Categorical(probs=probs).sample().numpy()

            next_state, reward, done, info = env.step(action)
            reward = -10. if done else reward
            trajectory.append((state, action, reward))  # 将(state, action, reward)三元组加入当前轨迹
            state = next_state

            if done:
                print("episode %4d, score %4d" % (episode_id, t))
                break

        T = len(trajectory)
        state, action, reward = [np.array(_) for _ in tuple(zip(*trajectory))]
        G = np.zeros(T)
        for t in reversed(range(T)):
            G[t] = (gamma * G[t + 1] if t < T - 1 else 0) + reward[t]

        with tf.GradientTape(persistent=True) as tape:
            b = tf.squeeze(baseline_network(state))
            probs = policy_network(state)
            log_prob = tf.math.log(tf.reduce_sum(
                tf.one_hot(action, depth=num_actions) * probs, axis=-1
            ))
            policy_loss = -tf.reduce_mean((G - b) * log_prob)
            baseline_loss = tf.reduce_mean(tf.square(G - b))
        grads = tape.gradient(policy_loss, policy_network.variables)
        optimizer.apply_gradients(zip(grads, policy_network.variables))
        grads = tape.gradient(baseline_loss, baseline_network.variables)
        optimizer.apply_gradients(zip(grads, baseline_network.variables))
