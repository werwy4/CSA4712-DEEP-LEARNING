# Experiment No: 74
# Title: Reinforcement Learning with Q-Network (DQN) for CartPole
# Aim: Train a Deep Q-Network agent to balance a pole on a cart
#      using experience replay and target networks.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib, collections

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ── 1. CartPole environment (custom lightweight simulation) ───────
# We replicate CartPole-v1 physics to avoid Gym dependency issues
class CartPoleEnv:
    """Simple CartPole-v1 physics simulation."""
    def __init__(self):
        self.gravity  = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length     = 0.5  # half-pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag  = 10.0
        self.tau        = 0.02
        self.theta_threshold = 12 * np.pi / 180
        self.x_threshold     = 2.4
        self.reset()

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4).astype(np.float32)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta); sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = ((self.gravity * sintheta - costheta * temp) /
                    (self.length * (4/3 - self.masspole * costheta**2 / self.total_mass)))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x         += self.tau * x_dot
        x_dot     += self.tau * xacc
        theta     += self.tau * theta_dot
        theta_dot += self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        done = bool(x < -self.x_threshold or x > self.x_threshold or
                    theta < -self.theta_threshold or theta > self.theta_threshold)
        reward = 0.0 if done else 1.0
        return self.state.copy(), reward, done

STATE_DIM  = 4
ACTION_DIM = 2

# ── 2. Build Q-network ────────────────────────────────────────────
def build_q_network():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(STATE_DIM,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(ACTION_DIM, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse')
    return model

q_network      = build_q_network()
target_network = build_q_network()
target_network.set_weights(q_network.get_weights())

# ── 3. Replay buffer ──────────────────────────────────────────────
replay_buffer = deque(maxlen=10000)

# ── 4. DQN Hyperparameters ────────────────────────────────────────
GAMMA         = 0.99
EPSILON       = 1.0
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE    = 64
TARGET_UPDATE = 10   # sync target network every N episodes

env = CartPoleEnv()

def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(ACTION_DIM)
    q_vals = q_network.predict(state[np.newaxis], verbose=0)[0]
    return np.argmax(q_vals)

def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states     = np.array([e[0] for e in batch], dtype=np.float32)
    actions    = np.array([e[1] for e in batch], dtype=np.int32)
    rewards    = np.array([e[2] for e in batch], dtype=np.float32)
    next_states= np.array([e[3] for e in batch], dtype=np.float32)
    dones      = np.array([e[4] for e in batch], dtype=np.float32)

    q_vals     = q_network.predict(states,      verbose=0)
    q_next     = target_network.predict(next_states, verbose=0)
    targets    = q_vals.copy()
    for i in range(BATCH_SIZE):
        targets[i, actions[i]] = (rewards[i] + GAMMA * np.max(q_next[i]) * (1 - dones[i]))

    history = q_network.fit(states, targets, batch_size=BATCH_SIZE, epochs=1, verbose=0)
    return history.history['loss'][0]

# ── 5. Training loop ──────────────────────────────────────────────
N_EPISODES  = 300
episode_rewards = []
episode_losses  = []

print("Training DQN on CartPole...")
for ep in range(N_EPISODES):
    state   = env.reset()
    total_r = 0
    losses  = []
    for _ in range(500):   # max steps
        action = select_action(state, EPSILON)
        next_state, reward, done = env.step(action)
        replay_buffer.append((state, action, reward, next_state, float(done)))
        state    = next_state
        total_r += reward
        loss     = train_step()
        if loss > 0:
            losses.append(loss)
        if done:
            break

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    episode_rewards.append(total_r)
    episode_losses.append(np.mean(losses) if losses else 0)

    if ep % TARGET_UPDATE == 0:
        target_network.set_weights(q_network.get_weights())

    if (ep + 1) % 50 == 0:
        avg_r = np.mean(episode_rewards[-50:])
        print(f"  Episode {ep+1:3d} | AvgReward(50): {avg_r:.1f} | ε={EPSILON:.3f}")

# ── 6. Plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Smooth rewards
window = 20
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
axes[0].plot(episode_rewards, alpha=0.3, color='steelblue', label='Raw')
axes[0].plot(range(window-1, len(episode_rewards)), smoothed,
             color='steelblue', lw=2, label=f'Smoothed (w={window})')
axes[0].axhline(195, color='red', linestyle='--', label='Solved (195)')
axes[0].set_title('DQN Episode Rewards'); axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(episode_losses, alpha=0.5, color='tomato')
smooth_l = np.convolve(episode_losses, np.ones(window)/window, mode='valid')
axes[1].plot(range(window-1, len(episode_losses)), smooth_l, color='tomato', lw=2)
axes[1].set_title('Q-Network Training Loss'); axes[1].set_xlabel('Episode')
axes[1].set_ylabel('MSE Loss'); axes[1].grid(True, alpha=0.3)

# Q-value landscape for a fixed state
x_range   = np.linspace(-2.4, 2.4, 50)
theta_range = np.linspace(-0.2, 0.2, 50)
XX, TT    = np.meshgrid(x_range, theta_range)
states_grid = np.c_[XX.ravel(), np.zeros(2500), TT.ravel(), np.zeros(2500)].astype(np.float32)
q_grid    = q_network.predict(states_grid, verbose=0)
max_q     = np.max(q_grid, axis=1).reshape(50, 50)
im = axes[2].contourf(XX, TT, max_q, levels=20, cmap='viridis')
plt.colorbar(im, ax=axes[2])
axes[2].set_xlabel('Cart Position'); axes[2].set_ylabel('Pole Angle (rad)')
axes[2].set_title('Q-Value Landscape (max Q over actions)')

plt.suptitle('DQN CartPole – Training Progress', fontsize=12)
plt.tight_layout()
plt.savefig('lab74_dqn_cartpole.png', dpi=100)
plt.show()

# ── 7. Evaluation run ──────────────────────────────────────────────
print("\n── DQN Evaluation (10 runs) ──")
eval_rewards = []
for _ in range(10):
    s = env.reset(); r = 0
    for _ in range(500):
        a = np.argmax(q_network.predict(s[np.newaxis], verbose=0)[0])
        s, rew, done = env.step(a)
        r += rew
        if done: break
    eval_rewards.append(r)
print(f"  Mean reward: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
print(f"  Max reward : {np.max(eval_rewards):.1f}")

# EXPECTED OUTPUT
# Episode rewards rise from ~20 to ~100-200 as training progresses.
# Loss decreases as Q-function approximation improves.
# Evaluation mean reward > 150 after 300 episodes.

# RESULT
# Deep Q-Networks combine Q-learning with neural function approximation.
# Experience replay decorrelates training samples, and target network
# stabilizes training. The agent learns a policy that balances the pole
# by learning the Q-value function for each state-action pair.
