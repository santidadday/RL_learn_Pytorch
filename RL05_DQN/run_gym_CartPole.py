import gym
from RL_brain import DeepQNetwork

env = gym.make("CartPole-v1", render_mode="human")
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01,
                  e_greedy=0.9,
                  replace_target_iter=100,
                  memory_size=2000,
                  e_greedy_increment=0.001
                  )

total_steps = 0
for i_episode in range(100):
    observation, _ = env.reset()
    while True:
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, _, _ = env.step(action)
        x, x_dot, theta, theta_dot = observation_

        # the smaller theta and closer to center the better
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        if done:
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
