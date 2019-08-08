import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import imageio
import sys
from Env import DeliveryEnvironment
sys.path.append("../")
from rl.agents.q_agent import QAgent

plt.style.use("seaborn-dark")


class DeliveryQAgent(QAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states_memory = []

    def act(self, s):

        # Get Q Vector
        q = np.copy(self.Q[s, :])

        # Avoid already visited states
        q[self.states_memory] = -np.inf

        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            possible_states = [x for x in range(self.actions_size) if x not in self.states_memory]

            a = np.random.choice(possible_states)

        return a

    def remember_state(self, s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []


class DeliveryAgentTimeWindow(DeliveryQAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_windows = []

    def act(self, s):

        # Get Q Vector
        q = np.copy(self.Q[s, :])

        # Avoid already visited states
        q[self.states_memory] = -np.inf

        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            available_states = [x for x in range(self.actions_size) if x not in self.states_memory]
            a = np.random.choice(available_states)

        return a


def run_episode(env, agent, verbose=1):
    s = env.reset()
    agent.reset_memory()

    max_step = env.n_stops

    episode_reward = 0

    i = 0
    while i < max_step:

        # Remember the states
        agent.remember_state(s)

        # Choose an action
        a = agent.act(s)

        # Take the action, and get the reward from environment
        s_next, r, done = env.step(a)

        # Tweak the reward
        r = -1 * r

        if verbose:
            print(s_next, r, done)

        # Update our knowledge in the Q-table
        agent.train(s, a, r, s_next)

        # Update the caches
        episode_reward += r
        s = s_next

        # If the episode is terminated
        i += 1
        if done:
            break

    return env, agent, episode_reward


def run_n_episodes(env, agent, name="training.gif", n_episodes=1000, render_each=10, fps=10):

    # Store the rewards
    rewards = []
    imgs = []

    # Experience replay
    for i in tqdm_notebook(range(n_episodes)):
        if i % 100 == 0:
            print('episod: ' + str(i))
        # Run the episode
        env, agent, episode_reward = run_episode(env, agent, verbose=0)
        rewards.append(episode_reward)
        
        if i % render_each == 0:
            img = env.render(return_img=True)
            imgs.append(img)

    # Show rewards
    plt.figure(figsize=(15, 3))
    plt.title("Rewards over training")
    plt.plot(rewards)
    plt.show()

    # Save imgs as gif
    imageio.mimsave(name, imgs, fps=fps)

    return env, agent
