# Experiment No: 76
# Title: Actor-Critic for Continuous Control
# Aim: Implement the Advantage Actor-Critic (A2C) algorithm for a
#      continuous pendulum swing-up task.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Pendulum environment (custom physics) ──────────────────────
class PendulumEnv:
    """
    Swing-up pendulum. State: [cos(θ), sin(θ), θ_dot].
    Action: torque in [-2, +2].
    Reward: -(θ² + 0.1·θ_dot² + 0.001·τ²)  (goal: upright θ=0).
    """
    def __init__(self):
        self.dt = 0.05; self.g = 10.0; self.m = 1.0; self.l = 1.0
        self.max_speed = 8.0; self.max_torque = 2.0

    def reset(self):
        self.theta    = np.random.uniform(-np.pi, np.pi)
        self.theta_dot = np.random.uniform(-1, 1)
        return self._state()

    def _state(self):
        return np.array([np.cos(self.theta), np.sin(self.theta),
                         self.theta_dot], dtype=np.float32)

    def step(self, action):
        u = np.clip(action, -self.max_torque, self.max_torque)
        cost = (self._angle_normalize(self.theta)**2
                + 0.1 * self.theta_dot**2 + 0.001 * u**2)
        self.theta_dot += ((-3*self.g/(2*self.l))*np.sin(self.theta+np.pi)
                           + 3/(self.m*self.l**2)*u) * self.dt
        self.theta += self.theta_dot * self.dt
        self.theta_dot = np.clip(self.theta_dot, -self.max_speed, self.max_speed)
        return self._state(), float(-cost), False   # continuous task

    @staticmethod
    def _angle_normalize(x):
        return ((x + np.pi) % (2*np.pi)) - np.pi

STATE_DIM  = 3
ACTION_DIM = 1    # single continuous torque

# ── 2. Actor and Critic networks ──────────────────────────────────
def build_actor():
    inp  = keras.Input(shape=(STATE_DIM,))
    x    = keras.layers.Dense(64, activation='relu')(inp)
    x    = keras.layers.Dense(64, activation='relu')(x)
    mu   = keras.layers.Dense(ACTION_DIM, activation='tanh')(x)   # mean
    log_std = keras.layers.Dense(ACTION_DIM, activation='linear')(x)
    model = keras.Model(inp, [mu, log_std], name='Actor')
    return model

def build_critic():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(STATE_DIM,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1,  activation='linear')
    ], name='Critic')
    return model

actor   = build_actor()
critic  = build_critic()
actor.summary()
critic.summary()

actor_opt  = keras.optimizers.Adam(1e-3)
critic_opt = keras.optimizers.Adam(1e-3)

GAMMA   = 0.99
ENV_MAX = 200     # steps per episode

# ── 3. Gaussian policy (continuous action) ────────────────────────
def select_action(state):
    s_tf = tf.constant(state[np.newaxis], dtype=tf.float32)
    mu, log_std = actor(s_tf, training=False)
    std    = tf.exp(tf.clip_by_value(log_std, -2, 2))
    action = mu + std * tf.random.normal(tf.shape(mu))
    action = tf.clip_by_value(action, -2.0, 2.0)
    return action.numpy()[0, 0]

def compute_log_prob(mu, log_std, action_tf):
    std = tf.exp(tf.clip_by_value(log_std, -2, 2))
    return (-0.5 * ((action_tf - mu) / (std + 1e-8))**2
            - tf.math.log(std + 1e-8) - 0.5 * np.log(2*np.pi))

# ── 4. A2C update ─────────────────────────────────────────────────
@tf.function
def a2c_update(states_tf, actions_tf, returns_tf):
    with tf.GradientTape() as ctape:
        values     = critic(states_tf, training=True)
        advantages = returns_tf - tf.squeeze(values)
        c_loss     = tf.reduce_mean(advantages**2)

    with tf.GradientTape() as atape:
        mu, log_std = actor(states_tf, training=True)
        log_probs   = compute_log_prob(mu, log_std, actions_tf)
        # Use advantages from critic (stop_gradient to not flow through critic)
        adv         = tf.stop_gradient(advantages)
        entropy     = 0.5 * (1 + tf.math.log(2*np.pi*(tf.exp(log_std))**2 + 1e-8))
        a_loss      = -tf.reduce_mean(log_probs * adv) - 0.01 * tf.reduce_mean(entropy)

    c_grads = ctape.gradient(c_loss, critic.trainable_variables)
    a_grads = atape.gradient(a_loss, actor.trainable_variables)
    critic_opt.apply_gradients(zip(c_grads, critic.trainable_variables))
    actor_opt.apply_gradients(zip(a_grads, actor.trainable_variables))
    return a_loss, c_loss

# ── 5. Training loop ──────────────────────────────────────────────
env = PendulumEnv()
N_EPISODES = 400
ep_rewards   = []
actor_losses = []
critic_losses= []

print("Training A2C on Pendulum...")
for ep in range(N_EPISODES):
    state   = env.reset()
    states, actions, rewards = [], [], []
    for _ in range(ENV_MAX):
        action = select_action(state)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append([action])
        rewards.append(reward)
        state = next_state
        if done:
            break

    # Compute discounted returns
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = np.array(returns, dtype=np.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    s_tf  = tf.constant(np.array(states, dtype=np.float32))
    a_tf  = tf.constant(np.array(actions, dtype=np.float32))
    r_tf  = tf.constant(returns[:, np.newaxis])

    al, cl = a2c_update(s_tf, a_tf, r_tf)
    ep_rewards.append(sum(rewards))
    actor_losses.append(al.numpy())
    critic_losses.append(cl.numpy())

    if (ep + 1) % 50 == 0:
        avg = np.mean(ep_rewards[-50:])
        print(f"  Ep {ep+1:3d} | AvgReward={avg:.1f}")

# ── 6. Plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
w = 20
smooth_r = np.convolve(ep_rewards, np.ones(w)/w, mode='valid')
axes[0].plot(ep_rewards, alpha=0.3, color='steelblue')
axes[0].plot(range(w-1, len(ep_rewards)), smooth_r, color='steelblue', lw=2)
axes[0].set_title('A2C Episode Rewards'); axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward'); axes[0].grid(True, alpha=0.3)

axes[1].plot(actor_losses,  alpha=0.5, label='Actor',  color='steelblue')
axes[1].plot(critic_losses, alpha=0.5, label='Critic', color='tomato')
axes[1].set_title('A2C Losses'); axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Policy visualization: torque vs angle
thetas = np.linspace(-np.pi, np.pi, 100)
states_vis = np.c_[np.cos(thetas), np.sin(thetas), np.zeros(100)].astype(np.float32)
mu_vis, _ = actor(states_vis, training=False)
torques   = (mu_vis.numpy().ravel() * 2.0)
axes[2].plot(thetas * 180/np.pi, torques, 'steelblue', lw=2)
axes[2].axhline(0, color='gray', linestyle='--')
axes[2].set_xlabel('Pendulum Angle (deg)'); axes[2].set_ylabel('Torque (action)')
axes[2].set_title('Learned Policy: Torque vs Angle')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Advantage Actor-Critic (A2C) – Pendulum Swing-Up', fontsize=12)
plt.tight_layout()
plt.savefig('lab76_a2c_pendulum.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Episode rewards rise from about -1000 to above -200 as the pendulum
# learns to stay upright. Actor and critic losses decrease over training.

# RESULT
# A2C uses a critic to estimate a baseline (state value V(s)) which
# reduces the variance of the policy gradient. The advantage A(s,a)=R-V(s)
# tells the actor whether its action was better or worse than expected.
# This allows stable learning of continuous control policies.
