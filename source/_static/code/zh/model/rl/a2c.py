import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym

num_episodes = 500
max_len_episode = 1000
learning_rate = 1e-3
gamma = 1.
num_actions = 2
num_envs = 32

global_policy_network = tf.keras.Sequential([
    tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=num_actions, activation=tf.nn.softmax),
])

policy_network = tf.keras.Sequential([
    tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=num_actions, activation=tf.nn.softmax),
])

critic_network = tf.keras.Sequential([
    tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=1)
])

if __name__ == '__main__':
    with open('a2c.py', 'r') as f:
        print(f.read())
    # https://stackoverflow.com/questions/60510441/implementing-a3c-on-tensorflow-2
    env = [gym.make('CartPole-v1') for _ in range(num_envs)]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    state = np.array([e.reset() for e in env])
    env_last_t, env_episode_id = [0] * num_envs, [0] * num_envs
    for t in range(max_len_episode * num_episodes):
        # env.render()
        probs = global_policy_network(state)
        action = tfp.distributions.Categorical(probs=probs).sample().numpy()

        results = [e.step(a) for e, a in zip(env, action)]
        next_state, reward, done, info = map(np.array, zip(*results))
        # reward = -10. if done else reward

        with tf.GradientTape(persistent=True) as tape:
            probs = policy_network(state)
            log_prob = tf.math.log(tf.reduce_sum(
                tf.one_hot(action, depth=num_actions) * probs, axis=-1
            ))
            v = tf.squeeze(critic_network(state))
            v_next = tf.squeeze(critic_network(next_state))
            v_ = reward + gamma * v_next * (1 - done)
            policy_loss = -tf.reduce_mean((v_ - v) * log_prob)
            critic_loss = huber_loss(v_, v)
        grads = tape.gradient(policy_loss, policy_network.variables)
        optimizer.apply_gradients(zip(grads, policy_network.variables))
        grads = tape.gradient(critic_loss, critic_network.variables)
        optimizer.apply_gradients(zip(grads, critic_network.variables))

        for i in range(num_envs):
            if done[i]:
                print("env %4d, episode %4d, score %4d" % (i, env_episode_id[i], t - env_last_t[i]))
                next_state[i] = env[i].reset()
                env_last_t[i], env_episode_id[i] = t, env_episode_id[i] + 1

        if t % 10 == 0:
            global_policy_network.set_weights([_.numpy() for _ in policy_network.variables])

        state = next_state
