import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from typing import Callable, NamedTuple, Optional, List
from dataclasses import dataclass
import gymnasium as gym
import itertools
import wandb
from functools import partial
import jax
import jax.numpy as jnp
import optax
import numpy as np
from collections import namedtuple, deque
import random
import os
import pickle
from flax.training.train_state import TrainState
from flax import struct
import csv
import flax
import flax.linen as nn

from utils import normalization, tree
from networks.MLP import sparse_init
from networks.value_networks import (
    DenseQNetwork,
    AtariQNetwork,
    MinAtarQNetwork,
    OctaxQNetwork,
)
from utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from utils.store_episode_returns_and_lengths import (
    StoreEpisodeReturnsAndLengths,
)
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import ale_py

gym.register_envs(ale_py)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "None"
    """the wandb's project name"""
    wandb_entity: str = "None"
    """the entity (team) of wandb's project"""
    exp_class: str = "tmp"
    """the class of the experiment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    num_checkpoints: int = 100
    """number of checkpoints to save during training"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    log_dir: str = "logs"
    """the logging directory"""
    env_type: str = "octax"
    """the type of environment"""

    # Algorithm specific arguments
    env_id: str = "filter"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    q_lr: float = 1e-4
    """the learning rate of the Q-network optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    start_epsilon: float = 1.0
    """the starting epsilon for exploration"""
    end_epsilon: float = 0.01
    """the ending epsilon for exploration"""
    explore_frac: float = 0.10
    """the fraction of total-timesteps for epsilon annealing"""
    h_lr_scale: float = 0.1
    """scale for auxiliary network learning rate"""
    lamda: float = 0.8
    """TD(lambda) parameter"""
    reg_coeff: float = 1.0
    """TDRC regularization parameter for auxiliary network"""
    sparse_init: float = 0.9
    """the sparsity of the neural network weights"""
    layer_norm: bool = True
    """whether to use layer normalization in the networks"""
    activation: str = "leaky_relu"
    """the activation function to use in the networks"""
    gradient_correction: bool = True
    """whether to use gradient correction (TDC vs GTD2)"""
    opt: str = "sgd"
    """optimizer to use"""
    net_arch: str = "minatar"
    """network architecture"""
    mlp_layers: Optional[List[int]] = None
    """MLP hidden layers"""
    periodic_checkpointing: bool = False
    """whether to save periodic checkpoints during training"""
    resume: bool = False
    """whether to resume from the last checkpoint if available"""
    use_spr: bool = False
    """whether to use SPR (Self-Predictive Representations)"""
    et_lr: float = 1e-4
    """learning rate for ET module"""
    et_feature_dim: int = 64
    """dimension of ET feature and trace space"""
    et_trace_coeff: float = 1.0
    """weight of ET trace consistency loss relative to Q prediction loss"""


class OctaxToGymAdapter(gym.Env):
    """Adapter to make Octax/Gymnax environment look like a standard Gym environment."""

    def __init__(self, env_id: str, seed: int = 0):
        try:
            from octax.environments import create_environment
            from octax.wrappers import OctaxGymnaxWrapper
        except ImportError:
            raise ImportError("Please install octax to use octax environments.")

        self.env_raw, self.metadata = create_environment(env_id)
        self.env = OctaxGymnaxWrapper(self.env_raw)
        self.env_params = self.env.default_params
        # Create a unique RNG key based on seed.
        # Note: If environment is destroyed and recreated with same seed, sequence repeats.
        self.rng = jax.random.PRNGKey(seed)
        self.env_state = None

        self.action_space = gym.spaces.Discrete(self.env.num_actions)

        # Get observation shape by running a reset
        rng_dummy = jax.random.PRNGKey(0)
        obs_dummy, _ = self.env.reset(rng_dummy, self.env_params)
        # Octax usually returns (C, H, W), we transpose to (H, W, C)
        self.obs_shape = (obs_dummy.shape[1], obs_dummy.shape[2], obs_dummy.shape[0])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.obs_shape, dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)

        self.rng, rng_reset = jax.random.split(self.rng)
        obs, self.env_state = self.env.reset(rng_reset, self.env_params)
        obs = jnp.transpose(obs, (1, 2, 0))
        return np.array(obs), {}

    def step(self, action):
        self.rng, rng_step = jax.random.split(self.rng)
        obs, self.env_state, reward, done, info = self.env.step(
            rng_step, self.env_state, action, self.env_params
        )
        obs = jnp.transpose(obs, (1, 2, 0))
        return np.array(obs), float(reward), bool(done), False, {}


def make_env(args, idx, run_name):
    if args.env_type == "atari":
        if args.capture_video and idx == 0:
            env = gym.make(args.env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(args.env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env = RecordEpisodeStatisticsJAX(env)
        # env = StoreEpisodeReturnsAndLengthsJAX(env)

        # Normalization must come after stats for correct results
        # if env_config.normalize_reward:
        env = ScaleRewardJAX(env, gamma=args.gamma)
        # if env_config.normalize_obs:
        env = NormalizeObservationJAX(env)

        env.action_space.seed(args.seed)
        return env
    elif args.env_type == "octax":
        print(f"Creating Octax environment: {args.env_id}")
        env = OctaxToGymAdapter(args.env_id, seed=args.seed)

        env = RecordEpisodeStatisticsJAX(env)
        env = ScaleRewardJAX(env, gamma=args.gamma)
        env = NormalizeObservationJAX(env)

        return env
    elif args.env_type == "minatar":
        print(f"Creating standard Gym Minatar environment: {args.env_id}")
        env = gym.make(args.env_id)

        # Episode statistics wrapper
        env = RecordEpisodeStatistics(env)
        env = StoreEpisodeReturnsAndLengths(env)

        # Normalization must come after stats for correct results
        env = normalization.ScaleReward(env, gamma=args.gamma)
        env = normalization.NormalizeObservation(env)

        return env


# TODO make this faster / compatible with jax.jit
class RecordEpisodeStatisticsJAX(gym.Wrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.episode_count = 0
        self.episode_start_times: jnp.ndarray = None
        self.episode_returns: Optional[jnp.ndarray] = None
        self.episode_lengths: Optional[jnp.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self.episode_start_times = jnp.full(
            self.num_envs, time.perf_counter(), dtype=jnp.float32
        )
        self.episode_returns = jnp.zeros(self.num_envs, dtype=jnp.float32)
        self.episode_lengths = jnp.zeros(self.num_envs, dtype=jnp.int32)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        dones = jnp.logical_or(terminations, truncations)
        num_dones = jnp.sum(dones)
        if num_dones:
            # if "episode" in infos or "_episode" in infos:
            #     raise ValueError(
            #         "Attempted to add episode stats when they already exist"
            #     )
            # else:
            #     infos["episode"] = {
            #         "r": jnp.where(dones, self.episode_returns, 0.0),
            #         "l": jnp.where(dones, self.episode_lengths, 0),
            #         "t": jnp.where(
            #             dones,
            #             jnp.round(time.perf_counter() - self.episode_start_times, 6),
            #             0.0,
            #         ),
            #     }
            #     if self.is_vector_env:
            #         infos["_episode"] = jnp.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones

            self.episode_lengths = self.episode_lengths.at[dones].set(0)
            self.episode_returns = self.episode_returns.at[dones].set(0)
            self.episode_start_times = self.episode_start_times.at[dones].set(
                time.perf_counter()
            )
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )


# TODO make this faster / compatible with jax.jit
class SampleMeanStdJAX:
    def __init__(self, shape=()):
        self.mean = jnp.zeros(shape, "float32")
        self.var = jnp.ones(shape, "float32")
        self.p = jnp.ones(shape, "float32")
        self.count = 0

    def update(self, x):
        if self.count == 0:
            self.mean = x
            self.p = jnp.zeros_like(x)
        self.mean, self.var, self.p, self.count = (
            self.update_mean_var_count_from_moments(
                self.mean, self.p, self.count, x * 1.0
            )
        )

    def update_mean_var_count_from_moments(self, mean, p, count, sample):
        new_count = count + 1
        new_mean = mean + (sample - mean) / new_count
        p = p + (sample - mean) * (sample - new_mean)
        new_var = 1 if new_count < 2 else p / (new_count - 1)
        return new_mean, new_var, p, new_count


# TODO make this faster / compatible with jax.jit
class NormalizeObservationJAX(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.obs_stats = SampleMeanStdJAX(shape=self.single_observation_space.shape)
        else:
            self.obs_stats = SampleMeanStdJAX(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(jnp.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(jnp.array([obs]))[0], info

    def normalize(self, obs):
        self.obs_stats.update(obs)
        return (obs - self.obs_stats.mean) / jnp.sqrt(self.obs_stats.var + self.epsilon)


# TODO make this faster / compatible with jax.jit
class ScaleRewardJAX(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    __doc__ = gym.core.Wrapper.__doc__

    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False
        self.reward_stats = SampleMeanStdJAX(shape=())
        self.reward_trace = jnp.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = jnp.array([rews])
        term = terminateds or truncateds
        self.reward_trace = self.reward_trace * self.gamma * (1 - term) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        self.reward_stats.update(self.reward_trace)
        return rews / jnp.sqrt(self.reward_stats.var + self.epsilon)


@struct.dataclass
class Config:
    d: dict = struct.field(pytree_node=False)

    def __getattribute__(self, name):
        d = object.__getattribute__(self, "d")
        try:
            return d[name]
        except KeyError:
            return object.__getattribute__(self, name)

    @classmethod
    def from_dict(cls: "Config", d: dict):
        return cls(d)

    @classmethod
    def from_args(cls: "Config", args: Args):
        return cls(vars(args))

    def to_dict(self, expand: bool = True):
        state_dict = flax_struct_to_dict(self)
        return transform_dict(state_dict, expand)


Agent = namedtuple("Agent", ["init_state", "step", "update"])


# ============================================================
# EXPECTED ELIGIBILITY TRACE (ET) MODULE
# ============================================================
class ETModule(nn.Module):
    """
    Implements the Expected Eligibility Trace as a learned function of state.

    The ET Bellman equation:
        z(s_t) = φ(s_t) + γλ · E_{s_{t-1} ~ p^{-1}(·|s_t)}[ z(s_{t-1}) ]

    Since we only observe ONE predecessor per step (streaming setting), the
    regression loss below averages over ALL observed predecessors over time —
    i.e., whenever s_t is visited from a different s_{t-1}, the network gets
    a new gradient signal.  At convergence z(s) → expected accumulated trace.

    This module also includes a Q-head so ET learning quality can be measured
    independently from the main QRC agent.
    """

    feature_dim: int   # dimension of φ(s) and z(s)
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # ── Encoder: obs → φ(s) ∈ R^{feature_dim} ──
        no_batch = x.ndim < 4  # single obs (no batch dim)
        if no_batch:
            x = x[None]
        x = x.reshape(x.shape[0], -1)          # flatten spatial dims
        phi = nn.Dense(128, name="enc_fc1")(x)
        phi = nn.relu(phi)
        phi = nn.Dense(self.feature_dim, name="enc_fc2")(phi)

        # ── Trace network: φ(s) → z(s) ∈ R^{feature_dim} ──
        # Trained to satisfy: z(s_t) ≈ φ(s_t) + γλ · z(s_{t-1})
        z = nn.Dense(self.feature_dim, name="trace_fc1")(phi)
        z = nn.relu(z)
        z = nn.Dense(self.feature_dim, name="trace_fc2")(z)

        # ── Q-head: z → Q_ET(s, ·) ∈ R^{action_dim} ──
        q_et = nn.Dense(self.action_dim, name="et_head")(z)

        if no_batch:
            phi = phi.squeeze(0)
            z = z.squeeze(0)
            q_et = q_et.squeeze(0)

        return phi, z, q_et


###################################################################################
class AgentState(NamedTuple):
    agent_config: Config
    train_state: TrainState
    h_state: TrainState
    h_trace: jnp.ndarray       # scalar trace for h
    grad_h_trace: jnp.ndarray  # gradient trace for h network
    grad_q_trace: jnp.ndarray  # gradient trace for Q network
    # ── ET additions ──
    et_state: TrainState       # ETModule (encoder + trace net + Q-head)
    prev_z: jnp.ndarray        # z(s_{t-1}) stored from the previous step


@partial(jax.jit, static_argnames=["action_dim"])
def agent_step(
    agent_state: AgentState,
    obs: jnp.ndarray,
    action_dim: int,
    epsilon: float,
    rng: jax.random.PRNGKey,
):
    params = agent_state.train_state.params
    q = agent_state.train_state.apply_fn(params, obs)
    argmax = jnp.argmax(q)

    rng, rng_e = jax.random.split(rng)

    def random_action_fn(rng_in):
        rng_out, rng_a = jax.random.split(rng_in)
        action = jax.random.randint(rng_a, shape=(), minval=0, maxval=action_dim)
        return action, rng_out

    def greedy_action_fn(rng_in):
        return argmax, rng_in

    action, rng_out = jax.lax.cond(
        jax.random.uniform(rng_e, minval=0.0, maxval=1.0) < epsilon,
        random_action_fn,
        greedy_action_fn,
        rng,
    )
    is_nongreedy = action != argmax
    return action, is_nongreedy, rng_out


def init_agent_state_qrc_agent(
    agent_config: Config, action_dim: int, obs_shape: tuple, rng: jax.random.PRNGKey
):
    net_kwargs = {
        "action_dim": action_dim,
        "layer_norm": agent_config.layer_norm,
        "activation": agent_config.activation,
        "kernel_init": sparse_init(sparsity=agent_config.sparse_init),
    }
    net_arch = agent_config.net_arch
    if net_arch == "mlp":
        net_kwargs["hiddens"] = agent_config.mlp_layers

    train_states = []
    lrs = [agent_config.q_lr, agent_config.h_lr_scale * agent_config.q_lr]
    # one network for q and one for h
    for net in range(2):
        if agent_config.env_type == "atari":
            network = AtariQNetwork(**net_kwargs)
            init_x = jnp.zeros(obs_shape)
        elif agent_config.env_type == "minatar":
            network = MinAtarQNetwork(**net_kwargs)
            init_x = jnp.zeros(obs_shape)
        elif agent_config.env_type == "octax":
            network = OctaxQNetwork(**net_kwargs)
            init_x = jnp.zeros(obs_shape)
        else:
            network = DenseQNetwork(**net_kwargs)
            init_x = jnp.zeros((obs_shape[0], np.prod(obs_shape[1:])))

        rng, _rng = jax.random.split(rng)
        params = network.init(_rng, init_x)

        tx = getattr(optax, agent_config.opt)(lrs[net])

        train_states.append(
            TrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=tx,
            )
        )

    def params_sum(params):
        return sum(
            jax.tree_util.tree_leaves(jax.tree.map(lambda x: np.prod(x.shape), params))
        )

    print(
        f"Total number of params: {params_sum(train_states[0].params) + params_sum(train_states[1].params)}"
    )
    grad_h_trace = jax.tree.map(jnp.zeros_like, train_states[0].params)
    grad_v_trace = jax.tree.map(jnp.zeros_like, train_states[1].params)
    h_trace = 0.0

    # ── Initialise ET module ──
    et_module = ETModule(
        feature_dim=agent_config.et_feature_dim,
        action_dim=action_dim,
    )
    rng, _rng = jax.random.split(rng)
    et_params = et_module.init(_rng, jnp.zeros(obs_shape))
    et_state = TrainState.create(
        apply_fn=et_module.apply,
        params=et_params,
        tx=getattr(optax, agent_config.opt)(agent_config.et_lr),
    )
    prev_z = jnp.zeros(agent_config.et_feature_dim)
    print(f"ET module params: {params_sum(et_params)}")

    return (
        AgentState(
            agent_config, *train_states,
            h_trace, grad_h_trace, grad_v_trace,
            et_state, prev_z,
        ),
        rng,
    )


def update_q_trace(e_tmins1, rho_t, gamma, lamda, grad):
    # e_{t} = rho_t*gamma*lamda*e_{t-1} + grad)
    e_t = jax.tree.map(lambda x, y: (rho_t * gamma * lamda * x) + y, e_tmins1, grad)
    return e_t


def reset_trace(trace):
    return tree.zeros(trace)


## Updates for QRC(λ) agent. Equations (26)-(28) in the paper
@partial(jax.jit, static_argnames=["terminated", "truncated", "is_nongreedy"])
def update_step_qrc_agent(agent_state, transition, terminated, truncated, is_nongreedy):
    obs, action, next_obs, reward = transition

    config = agent_state.agent_config
    train_state = agent_state.train_state
    q_params = train_state.params
    h_params = agent_state.h_state.params
    h_tm1 = agent_state.h_trace
    grad_h_trace_tm1 = agent_state.grad_h_trace
    grad_q_trace_tm1 = agent_state.grad_q_trace

    def get_q(params):
        q = train_state.apply_fn(params, obs)
        return q[action]

    q_grads = jax.grad(get_q)(q_params)

    def get_td_error(params):
        q = train_state.apply_fn(params, obs)
        q_taken = q[action]
        next_q_vect = train_state.apply_fn(params, next_obs)
        td_error = (
            (config.gamma * jnp.max(next_q_vect, axis=-1)) * (1 - terminated)
            + reward
            - q_taken
        )
        return td_error.squeeze(), q_taken  # Ensure scalar output

    (td_error, q_val), td_error_grad = jax.value_and_grad(get_td_error, has_aux=True)(
        q_params
    )

    def get_h(params):
        h = agent_state.h_state.apply_fn(params, obs)
        return h[action]

    h_t, h_grads = jax.value_and_grad(get_h)(h_params)
    rho_t = 1.0  # rho here can either be 1 when a greedy action is taken or zero for non greedy action, and we simply just cut the traces if a non-greedy action is taken. i.e, when rho = 0.
    h_trace_t = (rho_t * config.gamma * config.lamda * h_tm1) + h_t
    grad_h_trace_t = update_q_trace(
        grad_h_trace_tm1, rho_t, config.gamma, config.lamda, h_grads
    )
    grad_q_trace_t = update_q_trace(
        grad_q_trace_tm1, rho_t, config.gamma, config.lamda, q_grads
    )

    # update q
    q_update = tree.scale(-h_trace_t, td_error_grad)  # GTD2 update: -trace(h) * ∇δ
    if config.gradient_correction:
        # TDC update: GTD2 + gradient correction
        q_update = tree.add(
            tree.scale(td_error, grad_q_trace_t),  # δ * trace(∇q)
            tree.scale(-h_t, q_grads),  # -h * ∇q
            q_update,
        )
    q_train_state = train_state.apply_gradients(
        grads=tree.neg(q_update)
    )  # Flip sign because Flax multiplies by -1

    # update h
    delta_z_h = tree.scale(td_error, grad_h_trace_t)
    h_h_grad = tree.scale(-h_t, h_grads)
    beta_params = tree.scale(-config.reg_coeff, h_params)

    h_update = jax.tree.map(
        lambda x, y, z: -(x + y + z), delta_z_h, h_h_grad, beta_params
    )

    h_train_state = agent_state.h_state.apply_gradients(grads=h_update)

    if terminated or truncated or is_nongreedy:
        h_trace_t = reset_trace(h_trace_t)
        grad_h_trace_t = reset_trace(grad_h_trace_t)
        grad_q_trace_t = reset_trace(grad_q_trace_t)

    # Calculate L2 norms for logging
    q_update_l2 = optax.global_norm(q_update)
    h_update_l2 = optax.global_norm(h_update)

    # ================================================================
    # ET UPDATE
    # The ET Bellman equation:
    #   z(s_t) = φ(s_t) + γλ · E[z(s_{t-1}) | s_t]
    #
    # We train the ETModule with two losses:
    #   1. Trace consistency: z(s_t) ≈ φ(s_t) + γλ · prev_z
    #      Over time, averaging over many (s_{t-1}, s_t) pairs gives
    #      the expected predecessor trace — the ET approximation.
    #   2. Q prediction: Q_ET(s_t, a_t) ≈ R + γ·max Q(s_{t+1})
    #      Measures whether ET features lead to good Q-value estimates.
    # ================================================================
    et_state = agent_state.et_state
    prev_z = agent_state.prev_z

    # Bootstrap target from the main Q-network (stop gradient — ET does not
    # affect the main QRC network; QRC and ET train independently here)
    next_q_main = jax.lax.stop_gradient(
        train_state.apply_fn(train_state.params, next_obs)
    )
    next_q_max = jnp.max(next_q_main)

    def et_loss_fn(et_params):
        phi_t, z_t, q_et = et_state.apply_fn(et_params, obs)

        # ── 1. Trace consistency loss (ET Bellman equation) ──
        # Target: stop_gradient so it doesn't "chase itself"
        trace_target = jax.lax.stop_gradient(
            phi_t + config.gamma * config.lamda * prev_z
        )
        trace_loss = jnp.mean((z_t - trace_target) ** 2)

        # ── 2. ET Q-prediction loss ──
        td_target_et = jax.lax.stop_gradient(
            reward + config.gamma * next_q_max * (1 - terminated)
        )
        et_q_loss = (td_target_et - q_et[action]) ** 2

        total = et_q_loss + config.et_trace_coeff * trace_loss
        return total, (z_t, q_et[action], td_target_et - q_et[action], trace_loss)

    (_, et_aux), et_grads = jax.value_and_grad(et_loss_fn, has_aux=True)(
        et_state.params
    )
    z_t, q_et_val, et_td_error, et_trace_loss = et_aux
    new_et_state = et_state.apply_gradients(grads=et_grads)

    # Update prev_z; reset on episode end / non-greedy (same rule as QRC traces)
    prev_z_new = jax.lax.stop_gradient(z_t)
    if terminated or truncated or is_nongreedy:
        prev_z_new = jnp.zeros_like(prev_z_new)

    metrics = {
        "td_error": td_error,
        "q_val": q_val,
        "h_val": h_t,
        "q_update_l2": q_update_l2,
        "h_update_l2": h_update_l2,
        "et_td_error": et_td_error,
        "et_q_val": q_et_val,
        "et_trace_loss": et_trace_loss,
    }

    return (
        AgentState(
            config,
            q_train_state,
            h_train_state,
            h_trace_t,
            grad_h_trace_t,
            grad_q_trace_t,
            new_et_state,
            prev_z_new,
        ),
        metrics,
    )


QRCAgent = Agent(init_agent_state_qrc_agent, agent_step, update_step_qrc_agent)


###################################################################################


def get_linear_epsilon_schedule(args: Args) -> Callable[[int], float]:
    start_epsilon = args.start_epsilon
    end_epsilon = args.end_epsilon
    assert 0.0 <= end_epsilon <= start_epsilon <= 1.0
    anneal_time = args.explore_frac * args.total_timesteps
    assert anneal_time > 0.0

    def epsilon_schedule(t: int) -> float:
        frac_annealed = min(t / anneal_time, 1.0)
        return (1.0 - frac_annealed) * start_epsilon + frac_annealed * end_epsilon

    return epsilon_schedule


def experiment(args: Args, agent: Agent, run_name: str):
    agent_config = Config.from_args(args)
    rng = jax.random.PRNGKey(args.seed)

    # Create and initialize the environment (single environment, not vectorized)
    env = make_env(args, 0, run_name)
    obs, _ = env.reset(seed=args.seed)
    obs = np.array(obs)  # Convert LazyFrames to numpy array
    episodes = 0
    episode_return = 0.0
    episode_length = 0

    # Initialize the agent
    action_dim = int(env.action_space.n)
    agent_state, rng = agent.init_state(agent_config, action_dim, obs.shape, rng)
    epsilon_schedule = get_linear_epsilon_schedule(args)

    # Tracking for logging
    start_time = time.time()
    last_log_step = 0

    # List to accumulate all log dictionaries
    all_logs = []
    checkpointing_frequency = args.total_timesteps // args.num_checkpoints

    start_step = 0
    if args.resume:
        checkpoint_path = os.path.join(args.log_dir, "checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)

            def restore_train_state(current, state_bytes):
                if current is None or state_bytes is None:
                    return current
                params = flax.serialization.from_bytes(
                    current.params, state_bytes["params"]
                )
                opt_state = flax.serialization.from_bytes(
                    current.opt_state, state_bytes["opt_state"]
                )
                return current.replace(
                    params=params, opt_state=opt_state, step=checkpoint_data["step"]
                )

            new_train_state = restore_train_state(
                agent_state.train_state, checkpoint_data["train_state"]
            )
            new_h_state = restore_train_state(
                agent_state.h_state, checkpoint_data["h_state"]
            )

            h_trace = flax.serialization.from_bytes(
                agent_state.h_trace, checkpoint_data["h_trace"]
            )
            grad_h_trace = flax.serialization.from_bytes(
                agent_state.grad_h_trace, checkpoint_data["grad_h_trace"]
            )
            grad_q_trace = flax.serialization.from_bytes(
                agent_state.grad_q_trace, checkpoint_data["grad_q_trace"]
            )

            # Restore ET state if present in checkpoint (backwards compat)
            new_et_state = agent_state.et_state
            restored_prev_z = agent_state.prev_z
            if "et_state" in checkpoint_data and checkpoint_data["et_state"] is not None:
                new_et_state = restore_train_state(agent_state.et_state, checkpoint_data["et_state"])
            if "prev_z" in checkpoint_data and checkpoint_data["prev_z"] is not None:
                restored_prev_z = flax.serialization.from_bytes(agent_state.prev_z, checkpoint_data["prev_z"])

            agent_state = AgentState(
                agent_config=agent_state.agent_config,
                train_state=new_train_state,
                h_state=new_h_state,
                h_trace=h_trace,
                grad_h_trace=grad_h_trace,
                grad_q_trace=grad_q_trace,
                et_state=new_et_state,
                prev_z=restored_prev_z,
            )

            start_step = checkpoint_data["step"] + 1
            rng = checkpoint_data["rng"]
            episodes = checkpoint_data["episodes"]
            episode_return = checkpoint_data["episode_return"]
            episode_length = checkpoint_data["episode_length"]
            print(f"Resumed at step {start_step}")

    for t in range(start_step, args.total_timesteps):
        epsilon = epsilon_schedule(t)
        action, is_nongreedy, rng = agent.step(
            agent_state, obs, action_dim, epsilon, rng
        )
        action = action.item()
        is_nongreedy = is_nongreedy.item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.array(next_obs)  # Convert LazyFrames to numpy array
        done = terminated or truncated

        episode_return += reward
        episode_length += 1

        # Handle final observation for truncated episodes
        real_next_obs = next_obs.copy()
        # if truncated:
        #     real_next_obs = np.array(
        #         info["final_observation"]
        #     )  # Convert to numpy array

        transition = (obs, action, real_next_obs, reward)

        agent_state, metrics = agent.update(
            agent_state, transition, terminated, truncated, is_nongreedy
        )

        if done:
            episodes += 1

            episode_return = 0.0
            episode_length = 0

            next_obs, info = env.reset()
            next_obs = np.array(next_obs)  # Convert LazyFrames to numpy array

        # Periodic logging every 1000 steps
        if t % 1000 == 0 and t > 0:
            steps_elapsed = t - last_log_step
            sps = int(steps_elapsed / (time.time() - start_time))

            if len(env.get_wrapper_attr("return_queue")) > 0:
                avg_return = np.mean(env.get_wrapper_attr("return_queue"))
                avg_length = np.mean(env.get_wrapper_attr("length_queue"))
            else:
                avg_return = 0.0
                avg_length = 0.0

            # print(
            #     f"Step: {t}, Avg Return: {avg_return:.2f}, Avg Length: {avg_length:.2f}, SPS: {sps}, Epsilon: {epsilon:.3f}"
            # )

            # Create log dictionary
            log_dict = {
                "global_step": t,
                "avg_return": avg_return,
                "avg_length": avg_length,
                "td_loss": float(metrics["td_error"]),
                "q_values": float(metrics["q_val"]),
                "h_values": float(metrics["h_val"]),
                "q_update_l2": float(metrics["q_update_l2"]),
                "h_update_l2": float(metrics["h_update_l2"]),
                "et_td_error": float(metrics["et_td_error"]),
                "et_q_val": float(metrics["et_q_val"]),
                "et_trace_loss": float(metrics["et_trace_loss"]),
                "SPS": sps,
                "epsilon": epsilon,
                "episodes": episodes,
            }

            wandb.log(
                {
                    # "global_step": t,
                    "charts/episodic_return": avg_return,
                    "charts/episodic_length": avg_length,
                    "charts/SPS": sps,
                    "charts/epsilon": epsilon,
                    "losses/td_loss": log_dict["td_loss"],
                    "losses/q_values": log_dict["q_values"],
                    "losses/h_values": log_dict["h_values"],
                    "updates/q_update_l2": log_dict["q_update_l2"],
                    "updates/h_update_l2": log_dict["h_update_l2"],
                    "et/td_error": log_dict["et_td_error"],
                    "et/q_val": log_dict["et_q_val"],
                    "et/trace_loss": log_dict["et_trace_loss"],
                    "episodes": episodes,
                },
                step=t,
            )

            # Accumulate log_dict for CSV export
            all_logs.append(log_dict.copy())

            last_log_step = t
            start_time = time.time()

        # Periodic model checkpointing AND RESUME CHECKPOINT
        if (t % checkpointing_frequency == 0 or t % 10000 == 0) and t > 0:

            def save_train_state(ts):
                if ts is None:
                    return None
                return {
                    "params": flax.serialization.to_bytes(ts.params),
                    "opt_state": flax.serialization.to_bytes(ts.opt_state),
                }

            checkpoint_data = {
                "step": t,
                "train_state": save_train_state(agent_state.train_state),
                "h_state": save_train_state(agent_state.h_state),
                "h_trace": flax.serialization.to_bytes(agent_state.h_trace),
                "grad_h_trace": flax.serialization.to_bytes(agent_state.grad_h_trace),
                "grad_q_trace": flax.serialization.to_bytes(agent_state.grad_q_trace),
                "et_state": save_train_state(agent_state.et_state),
                "prev_z": flax.serialization.to_bytes(agent_state.prev_z),
                "rng": rng,
                "episodes": episodes,
                "episode_return": episode_return,
                "episode_length": episode_length,
                "wandb_run_id": wandb.run.id if wandb.run else None,
            }

            checkpoint_path = os.path.join(args.log_dir, "checkpoint.pkl")
            tmp_path = checkpoint_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            os.replace(tmp_path, checkpoint_path)
            # print(f"Resume checkpoint saved to {checkpoint_path}")

        # Periodic model checkpointing
        if (
            args.periodic_checkpointing
            and t % checkpointing_frequency == 0
            and t > 0
            and args.save_model
        ):
            model_path = f"{args.log_dir}/checkpoint_{t}.cleanrl_model"
            with open(model_path, "wb") as f:
                f.write(
                    flax.serialization.to_bytes(
                        [agent_state.train_state.params, agent_state.h_state.params]
                    )
                )
            print(f"Checkpoint saved to {model_path}")

        obs = next_obs

    # Final model saving
    if args.save_model:
        model_path = f"{args.log_dir}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [agent_state.train_state.params, agent_state.h_state.params]
                )
            )
        print(f"Final model saved to {model_path}")

    # Save logs to CSV
    if all_logs:
        csv_path = os.path.join(args.log_dir, "training_data.csv")
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = all_logs[0].keys()
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(all_logs)
        print(f"Logs saved to: {csv_path}")

    return env


def define_metrics():
    wandb.define_metric("global_step")
    wandb.define_metric("episodes")
    wandb.define_metric("charts/*", step_metric="global_step")


def main(
    experiment: Callable,
    agent: Agent,
    define_metrics: Callable[[None], None],
):
    import tyro

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"

    args.log_dir = os.path.join(args.log_dir, run_name)
    # Check if SCRATCH environment variable exists
    if "SCRATCH" in os.environ:
        args.log_dir = os.path.join(os.environ["SCRATCH"], "streamRepL", args.log_dir)
        print(f"Using SCRATCH directory for logs: {args.log_dir}")
    os.makedirs(args.log_dir, exist_ok=True)

    wandb_id = wandb.util.generate_id()
    if args.resume:
        checkpoint_path = os.path.join(args.log_dir, "checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint to resume: {checkpoint_path}")
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
            wandb_id = checkpoint_data.get("wandb_run_id")

    # Save config to JSON
    import json

    config_path = os.path.join(args.log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to: {config_path}")

    print(f"Run name: {run_name}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=False,
            id=wandb_id,
            resume="allow",
        )
    else:
        wandb.init(mode="disabled")

    define_metrics()

    start_time = time.time()
    env = jax.block_until_ready(experiment(args, agent, run_name))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time / 60:.2f} minutes")
    total_steps = args.total_timesteps
    wandb.run.summary["SPS"] = int(total_steps / elapsed_time)
    wandb.finish()


if __name__ == "__main__":
    main(
        experiment,
        QRCAgent,
        define_metrics,
    )
