import threading
from typing import Mapping, Sequence
from queue import Queue
import os
import time
from agent.agent_torch import  ReplayBuffer, ReplayDataset
from agent.distributional_torch import QuantileDiscreteValuedHead
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from environment.Environment import TradingEnv
from agent.agent_torch import D4PG
from Utils import Utils

from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_integer('train_sim', 3_0000, 'train episodes (Default 40_000)')
flags.DEFINE_integer('eval_sim', 5_000, 'evaluation episodes (Default 40_000)')
flags.DEFINE_integer('init_ttm', 30, 'number of days in one episode (Default 60)')
flags.DEFINE_float('mu', 0.0, 'spot drift (Default 0.2)')
flags.DEFINE_integer('n_step', 5, 'DRL TD Nstep (Default 5)')
flags.DEFINE_float('init_vol', 0.3, 'initial spot vol (Default 0.2)')
flags.DEFINE_float('poisson_rate', 1.0, 'possion rate of new optiosn in liability portfolio (Default 1.0)')
flags.DEFINE_float('moneyness_mean', 1.0, 'new optiosn moneyness mean (Default 1.0)')
flags.DEFINE_float('moneyness_std', 0.0, 'new optiosn moneyness std (Default 0.0)')
flags.DEFINE_string('critic', 'qr-huber', 'critic distribution type - c51, qr-huber, qr-gl, qr-gl_tl, '
                                     'qr-lapl, qr-lapl_tl, iqn-huber (Default c51)')
flags.DEFINE_float('spread', 0.02, 'Hedging transaction cost (Default 0.0)')
flags.DEFINE_string('obj_func', 'meanvar', 'Objective function select from meanstd, weightedmean, meanvar, var or cvar (Default var)')
flags.DEFINE_float('std_coef', 1.645, 'Std coefficient when obj_func=meanstd. (Default 1.645)')
flags.DEFINE_float('threshold', 0.95, 'Objective function threshold. (Default 0.95)')
flags.DEFINE_float('quantile_interval', 0.01, 'interval of quantiles . (Default 0.01)')
flags.DEFINE_boolean('even_interval', True, 'evenly split the Z quantile interval or put more weight on left tail (Default True)')
flags.DEFINE_float('vov', 0.5, 'Vol of vol, zero means BSM; non-zero means SABR (Default 0.0)')
flags.DEFINE_list('liab_ttms', ['60', ], 'List of maturities selected for new adding option (Default [60,])')
flags.DEFINE_integer('hed_ttm', 30, 'Hedging option maturity in days (Default 20)')
flags.DEFINE_list('action_space', ['0', '3'], 'Hedging action space (Default [0,3])')
flags.DEFINE_string('logger_prefix', 'logs', 'Prefix folder for logger (Default logs)')
flags.DEFINE_string('agent_path', '', 'trained agent path, only used when eval_only=True')
flags.DEFINE_float('actor_lr', 1e-5, 'Learning rate for actor optimizer (Default 1e-4)')
flags.DEFINE_float('critic_lr', 1e-3, 'Learning rate for critic optimizer (Default 1e-4)')
flags.DEFINE_integer('batch_size', 256, 'Batch size to train the Network (Default 256)')
flags.DEFINE_boolean('vega_obs', True,
                     'Include portfolio vega and hedging option vega in state variables (Default False)')
flags.DEFINE_integer('eval_seed', 8865, 'Evaluation Seed (Default 4321)')
flags.DEFINE_boolean('gbm', False, 'GBM (Default False)')
flags.DEFINE_boolean('sabr', True, 'SABR (Default False)')
flags.DEFINE_float('discount', 0.99, 'reward discount rate (Default 0.99)')
flags.DEFINE_boolean('stress_test', False, 'Stress test indicator (Default False)')




class CriticMultiplexer(nn.Module):
    """Concatenate obs and action."""
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        obs, action = inputs
        return torch.cat([obs, action], dim=-1)


class LayerNormMLP(nn.Module):
    def __init__(self, layer_sizes, activate_final=True):
        """
        layer_sizes: list of ints, e.g. [512,512,256]
        """
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.LayerNorm(layer_sizes[i+1]))
            layers.append(nn.ReLU())
        self.activate_final = activate_final
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        return F.relu(out) if self.activate_final else out

class QuantileValueHead(nn.Module):
    def __init__(self, in_dim, num_quantiles):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_quantiles)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)   # (B, Kq)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, layer_sizes: Sequence[int] = (256, 256, 256)):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for size in layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.ReLU())
            input_dim = size
        layers.append(nn.Linear(input_dim, action_dim))

        self.model = nn.Sequential(*layers)

        # initialize
        # small init on last layer weights to avoid huge outputs at start
        nn.init.uniform_(self.model[-1].weight, -3e-3, 3e-3)
        if self.model[-1].bias is not None:
            nn.init.zeros_(self.model[-1].bias)

    def forward(self, obs):
        """
        :return:  action in (-1,1)
        """
        raw = self.model(obs)
        return torch.tanh(raw)

# class CriticNetwork(nn.Module):
#     def __init__(self, obs_dim, action_dim, quantile_interval: float = 0.001, layer_sizes: Sequence[int] = (512, 512, 256)):
#         super().__init__()
#         self.quantiles = np.arange(quantile_interval, 1.0, quantile_interval)
#         self.num_quantiles = len(self.quantiles)
#
#         input_dim = obs_dim + action_dim
#         layers = []
#         for size in layer_sizes:
#             layers.append(nn.Linear(input_dim, size))
#             layers.append(nn.LayerNorm(size))
#             layers.append(nn.ReLU())
#             input_dim = size
#         layers.append(nn.Linear(input_dim, self.num_quantiles))  # 输出 quantile 值
#         self.model = nn.Sequential(*layers)
#         self.quantile_head = QuantileDiscreteValuedHead(self.quantiles)
#
#     def forward(self, obs, action):
#         x = torch.cat([obs, action], dim=-1)
#         q_values = self.model(x)
#         return self.quantile_head(q_values)
class CriticNetwork(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        mlp_hidden_sizes,      # e.g. [512,512,256]
        quantiles,             # ndarray of positions

    ):
        super().__init__()

        # 1. Multiplexer
        self.mux = CriticMultiplexer()

        # 2. Feature extractor (LayerNorm MLP)
        input_dim = obs_dim + action_dim
        mlp_layer_sizes = [input_dim] + list(mlp_hidden_sizes)
        self.mlp = LayerNormMLP(mlp_layer_sizes, activate_final=True)

        # 3. Final critic head producing quantile values
        self.value_head = QuantileValueHead(
            in_dim=mlp_hidden_sizes[-1],
            num_quantiles=len(quantiles)
        )

        # 4. Your quantile distribution wrapper head
        self.dist_head = QuantileDiscreteValuedHead(
            quantiles=quantiles,
        )

    def forward(self, obs, action):
        x = self.mux((obs, action))
        h = self.mlp(x)
        quantile_values = self.value_head(h)             # (B, Kq)
        qdist = self.dist_head(quantile_values)          # QuantileDistribution
        return qdist

def make_quantile_networks(action_space, observation_space, policy_layer_sizes=(256,256,256), critic_layer_sizes=(512,512,256), quantile_interval=0.01,even_interval = True):
    """
    :return: dict: {'observation': nn.Module, 'policy': nn.Module, 'critic': nn.Module}
    """
    obs_dim = np.prod(observation_space.shape)
    action_dim = np.prod(action_space.shape)
    quantiles = np.arange(quantile_interval, 1.0, quantile_interval)
    n_quantiles = len(quantiles)
    if not even_interval:
        n_left = int(n_quantiles * 0.333)
        n_mid = int(n_quantiles * 0.333)
        n_right = n_quantiles - n_left - n_mid

        left = np.linspace(quantile_interval, 0.2, n_left, endpoint=False)
        mid = np.linspace(0.2, 0.5, n_mid, endpoint=False)
        right = np.linspace(0.5, 0.99999, n_right, endpoint=True)

        quantiles = np.concatenate([left, mid, right])

        if len(quantiles) > n_quantiles:
            quantiles = quantiles[:n_quantiles]
        elif len(quantiles) < n_quantiles:
            quantiles = np.concatenate([quantiles, np.full(n_quantiles - len(quantiles), quantiles[-1])])

    observation_network = nn.Identity()
    policy_network = nn.Sequential(
        observation_network,
        PolicyNetwork(obs_dim, action_dim, policy_layer_sizes)
    )
    # critic_network = CriticNetwork(obs_dim, action_dim, quantile_interval, critic_layer_sizes)
    critic_network = CriticNetwork(obs_dim, action_dim, critic_layer_sizes, quantiles)

    return {
        'observation': observation_network,
        'policy': policy_network,
        'critic': critic_network,
    }

class StateNormalizer:
    def __init__(self, window_len=100, vega_obs=False):
        self.window_len = window_len
        self.vega_obs = vega_obs
        self.price_memory = []
        self.gamma_memory = []
        if self.vega_obs:
            self.vega_memory = []

    def normalize(self, state):
        # state[0]: action, state[1]: price, state[2]: gamma, state[3]: vega (optional)

        # --- normalize price using sliding window ---
        price = state[1]
        self.price_memory.append(price)
        if len(self.price_memory) == 1:
            mu_price, std_price = 100.0, 1.0
        else:
            recent_prices = self.price_memory[-self.window_len :]
            mu_price, std_price = np.mean(recent_prices), np.std(recent_prices)
            std_price = max(std_price, 1e-3)
        norm_price = (price - mu_price) / std_price

        # --- normalize gamma using sliding window ---
        gamma = state[2]
        self.gamma_memory.append(gamma)
        if len(self.gamma_memory) == 1:
            mu_gamma, std_gamma = 0.0, 1.0
        else:
            recent_gamma = self.gamma_memory[-self.window_len :]
            mu_gamma, std_gamma = np.mean(recent_gamma), np.std(recent_gamma)
            std_gamma = max(std_gamma, 1e-3)
        norm_gamma = (gamma - mu_gamma) / std_gamma

        # --- normalize action ---
        # 如果动作是 [-1,1]，映射到 [0,1]
        norm_action = (state[0] + 1.0) * 0.5

        # --- normalize vega if exists ---
        if self.vega_obs:
            vega = state[3]
            self.vega_memory.append(vega)
            if len(self.vega_memory) == 1:
                mu_vega, std_vega = 0.0, 1.0
            else:
                recent_vega = self.vega_memory[-self.window_len :]
                mu_vega, std_vega = np.mean(recent_vega), np.std(recent_vega)
                std_vega = max(std_vega, 1e-3)
            norm_vega = (vega - mu_vega) / std_vega
            return np.array([norm_action, norm_price, norm_gamma, norm_vega], dtype=np.float32)

        return np.array([norm_action, norm_price, norm_gamma], dtype=np.float32)


def main(argv):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    work_folder = f"{FLAGS.logger_prefix}/spread={FLAGS.spread}_obj={FLAGS.obj_func}_critic={FLAGS.critic}_{timestamp}"
    os.makedirs(work_folder, exist_ok=True)
    checkpoint_dir = os.path.join(work_folder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(work_folder, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"📁 Working directory: {work_folder}")

    # Create an environment, grab the spec, and use it to create networks.
    utils = Utils(init_ttm=FLAGS.init_ttm, np_seed=1561, num_sim=FLAGS.train_sim, spread=FLAGS.spread, volvol=FLAGS.vov,
                  sabr=FLAGS.sabr, gbm=FLAGS.gbm, hed_ttm=FLAGS.hed_ttm,
                  init_vol=FLAGS.init_vol, poisson_rate=FLAGS.poisson_rate,
                  moneyness_mean=FLAGS.moneyness_mean, moneyness_std=FLAGS.moneyness_std,
                  mu=FLAGS.mu, ttms=[int(ttm) for ttm in FLAGS.liab_ttms],
                  action_low=float(FLAGS.action_space[0]), action_high=float(FLAGS.action_space[1]))

    environment = TradingEnv(utils=utils)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    agent_networks = make_quantile_networks(action_space=environment.action_space,observation_space=environment.observation_space,
                                            quantile_interval=FLAGS.quantile_interval, even_interval= FLAGS.even_interval)

    # Construct the agent for both actor and learner.
    agent = D4PG(
        obj_func=FLAGS.obj_func,
        threshold=FLAGS.threshold,
        critic_loss_type=FLAGS.critic,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network = agent_networks['observation'],
        n_step=FLAGS.n_step,
        discount=1.0,
        sigma=0.3,  # pytype: disable=wrong-arg-types
        quantile_interval = FLAGS.quantile_interval,
        actor_lr= FLAGS.actor_lr,
        critic_lr=FLAGS.critic_lr,
        device = device,
        ckpt_dir = checkpoint_dir,
        log_dir=log_dir,
    )

    # create replay buffer
    buffer = ReplayBuffer(
        obs_shape=environment.observation_space.shape,
        action_shape=environment.action_space.shape,
        capacity=100000,
        batch_size=FLAGS.batch_size,
        n_step=FLAGS.n_step,
        gamma=FLAGS.discount,
        device=device
    )
    dataset = ReplayDataset(buffer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=0
    )
    dataset_iterator = iter(dataloader)


    #start distributional training
    stop_event = threading.Event()
    sync_queue = Queue()



    actor_thread = threading.Thread(
        target=agent.actor_loop,
        args=(environment, buffer, sync_queue, stop_event),
        daemon=True
    )
    actor_thread.start()

    agent.learner_loop(dataset_iterator, sync_queue, stop_event, FLAGS.train_sim)
    agent.save_rms(os.path.join(checkpoint_dir, "rms_state.pth"))
    stop_event.set()
    actor_thread.join()


if __name__ == '__main__':
    app.run(main)