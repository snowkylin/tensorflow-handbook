import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym

num_episodes = 500
max_len_episode = 1000
learning_rate = 1e-2
gamma = 1.
num_actions = 2
net_type = 'shared'

if net_type == 'individual':
    policy_network = tf.keras.Sequential([
        tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=num_actions, activation=tf.nn.softmax),
    ])

    critic_network = tf.keras.Sequential([
        tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=1),
    ])
elif net_type == 'shared':
    # in https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
    input_layer = tf.keras.layers.Input(shape=(4))
    common_layer = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
    policy_layer = tf.keras.layers.Dense(units=num_actions, activation=tf.nn.softmax)
    critic_layer = tf.keras.layers.Dense(units=1)
    policy_network = tf.keras.Model(inputs=input_layer, outputs=policy_layer(common_layer(input_layer)))
    critic_network = tf.keras.Model(inputs=input_layer, outputs=critic_layer(common_layer(input_layer)))

if __name__ == '__main__':
    with open('actor_critic.py', 'r') as f:
        print(f.read())
    env = gym.make('CartPole-v1')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    for episode_id in range(num_episodes):
        state = env.reset()
        for t in range(max_len_episode):
            env.render()
            probs = policy_network(np.expand_dims(state, axis=0))[0]
            action = tfp.distributions.Categorical(probs=probs).sample().numpy()

            next_state, reward, done, info = env.step(action)
            reward = -10 if done else reward

            with tf.GradientTape(persistent=True) as tape:
                probs = policy_network(np.expand_dims(state, axis=0))[0]
                log_prob = tf.math.log(tf.reduce_sum(
                    tf.one_hot(action, depth=num_actions) * probs, axis=-1
                ))
                v = critic_network(np.expand_dims(state, axis=0))[0]
                v_next = critic_network(np.expand_dims(next_state, axis=0))[0, 0]
                v_ = reward + gamma * v_next * (1 - done)
                policy_loss = -(v_ - v) * log_prob
                critic_loss = huber_loss(v_, v)
                # critic_loss = tf.reduce_mean(tf.square(delta))
            grads = tape.gradient(policy_loss, policy_network.variables)
            optimizer.apply_gradients(zip(grads, policy_network.variables))
            grads = tape.gradient(critic_loss, critic_network.variables)
            optimizer.apply_gradients(zip(grads, critic_network.variables))

            if done:
                print("episode %4d, score %4d" % (episode_id, t))
                break

            state = next_state
