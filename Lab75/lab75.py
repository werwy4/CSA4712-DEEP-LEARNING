# Experiment No: 75
# Title: Policy Gradient REINFORCE for Grid Navigation
# Aim: Implement the REINFORCE policy-gradient algorithm to learn
#      a navigation policy in a custom grid-world environment.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Grid-world Environment ─────────────────────────────────────
class GridWorld:
    """
    5×5 grid. Agent starts at (0,0), goal at (4,4).
    Actions: 0=Up, 1=Down, 2=Left, 3=Right.
    Reward: +10 at goal, −0.1 per step, −1 for hitting wall.
    """
    SIZE    = 5
    GOAL    = (4, 4)
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def reset(self):
        self.pos = [0, 0]
        return self._state()

    def _state(self):
        s = np.zeros(self.SIZE * self.SIZE, dtype=np.float32)
        s[self.pos[0] * self.SIZE + self.pos[1]] = 1.0
        return s

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        new_r  = self.pos[0] + dr
        new_c  = self.pos[1] + dc
        if 0 <= new_r < self.SIZE and 0 <= new_c < self.SIZE:
            self.pos = [new_r, new_c]
            reward = -0.1
        else:
            reward = -1.0   # wall penalty
        done = (self.pos[0] == self.GOAL[0] and self.pos[1] == self.GOAL[1])
        if done:
            reward = 10.0
        return self._state(), reward, done

STATE_DIM  = 25   # 5×5 one-hot
ACTION_DIM = 4

# ── 2. Policy network ─────────────────────────────────────────────
policy_net = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(STATE_DIM,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(ACTION_DIM, activation='softmax')
], name='PolicyNetwork')
policy_net.summary()

optimizer = keras.optimizers.Adam(learning_rate=5e-3)

# ── 3. REINFORCE utility functions ────────────────────────────────
GAMMA = 0.99

def compute_returns(rewards, gamma):
    """Compute discounted returns G_t."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns, dtype=np.float32)
    # Normalize
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def run_episode(env, max_steps=200):
    state = env.reset()
    states, actions, rewards = [], [], []
    for _ in range(max_steps):
        probs  = policy_net(state[np.newaxis], training=False).numpy()[0]
        action = np.random.choice(ACTION_DIM, p=probs)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        if done:
            break
    return states, actions, rewards

@tf.function
def update_policy(states_tf, actions_tf, returns_tf):
    with tf.GradientTape() as tape:
        probs    = policy_net(states_tf, training=True)
        log_prob = tf.reduce_sum(
            tf.one_hot(actions_tf, ACTION_DIM) * tf.math.log(probs + 1e-8),
            axis=1)
        loss = -tf.reduce_mean(log_prob * returns_tf)
    grads = tape.gradient(loss, policy_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
    return loss

# ── 4. Training loop ──────────────────────────────────────────────
env = GridWorld()
N_EPISODES = 1000
ep_rewards = []
ep_lengths = []
ep_losses  = []
goal_ep    = []   # episodes where goal was reached

print("Training REINFORCE...")
for ep in range(N_EPISODES):
    states, actions, rewards = run_episode(env)
    total_r = sum(rewards)
    returns = compute_returns(rewards, GAMMA)

    states_tf  = tf.constant(np.array(states,  dtype=np.float32))
    actions_tf = tf.constant(np.array(actions, dtype=np.int32))
    returns_tf = tf.constant(returns)

    loss = update_policy(states_tf, actions_tf, returns_tf).numpy()

    ep_rewards.append(total_r)
    ep_lengths.append(len(rewards))
    ep_losses.append(loss)
    if rewards[-1] == 10.0:
        goal_ep.append(ep)

    if (ep + 1) % 100 == 0:
        avg_r = np.mean(ep_rewards[-100:])
        goal_rate = sum(1 for e in goal_ep if e >= ep - 99) / 100 * 100
        print(f"  Ep {ep+1:4d} | AvgReward={avg_r:.2f} | GoalRate={goal_rate:.0f}%")

# ── 5. Plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

w = 50
smooth_r = np.convolve(ep_rewards, np.ones(w)/w, mode='valid')
axes[0].plot(ep_rewards, alpha=0.3, color='steelblue')
axes[0].plot(range(w-1, len(ep_rewards)), smooth_r, color='steelblue', lw=2)
axes[0].set_title('Episode Rewards'); axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward'); axes[0].grid(True, alpha=0.3)

smooth_l = np.convolve(ep_losses, np.ones(w)/w, mode='valid')
axes[1].plot(ep_losses, alpha=0.3, color='tomato')
axes[1].plot(range(w-1, len(ep_losses)), smooth_l, color='tomato', lw=2)
axes[1].set_title('Policy Loss'); axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Loss'); axes[1].grid(True, alpha=0.3)

# Policy heatmap: max probability action at each cell
grid_probs = np.zeros((GridWorld.SIZE, GridWorld.SIZE), dtype=np.float32)
action_grid= np.zeros((GridWorld.SIZE, GridWorld.SIZE), dtype=int)
arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
for r in range(GridWorld.SIZE):
    for c in range(GridWorld.SIZE):
        s = np.zeros(STATE_DIM, dtype=np.float32)
        s[r * GridWorld.SIZE + c] = 1.0
        probs = policy_net(s[np.newaxis], training=False).numpy()[0]
        grid_probs[r, c]  = np.max(probs)
        action_grid[r, c] = np.argmax(probs)

cmap = plt.cm.YlGn
im = axes[2].imshow(grid_probs, cmap=cmap, vmin=0, vmax=1)
for r in range(GridWorld.SIZE):
    for c in range(GridWorld.SIZE):
        a = arrows[action_grid[r, c]]
        axes[2].text(c, r, a, ha='center', va='center', fontsize=14, fontweight='bold')
axes[2].text(GridWorld.GOAL[1], GridWorld.GOAL[0], 'G',
             ha='center', va='center', fontsize=14, color='red', fontweight='bold')
plt.colorbar(im, ax=axes[2])
axes[2].set_title('Learned Policy (action with max probability)')
axes[2].set_xticks(range(GridWorld.SIZE))
axes[2].set_yticks(range(GridWorld.SIZE))

plt.suptitle('REINFORCE Policy Gradient – Grid Navigation', fontsize=12)
plt.tight_layout()
plt.savefig('lab75_reinforce.png', dpi=100)
plt.show()

# ── 6. Demo run ────────────────────────────────────────────────────
print("\n── Demo Episode ──")
env2 = GridWorld()
state = env2.reset()
path  = [tuple(env2.pos)]
for step in range(30):
    probs  = policy_net(state[np.newaxis], training=False).numpy()[0]
    action = np.argmax(probs)
    state, reward, done = env2.step(action)
    path.append(tuple(env2.pos))
    if done:
        print(f"  Reached goal in {step+1} steps!")
        break

print("  Path:", ' → '.join(str(p) for p in path))

# EXPECTED OUTPUT
# After ~600-800 episodes, the agent reliably reaches the goal.
# Policy heatmap shows arrows pointing towards (4,4).
# Average reward increases from negative to near +10.

# RESULT
# REINFORCE updates the policy in the direction that maximises expected
# discounted return. It directly optimises the policy without a value
# function. Normalised returns reduce variance and improve convergence.
# The learned policy navigates from (0,0) to (4,4) in minimal steps.
