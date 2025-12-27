import copy
import dataclasses
from typing import Iterator, List, Optional, Tuple
from agent.learning import D4PGLearner
import threading
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from absl import flags
import random
import time
import os
FLAGS = flags.FLAGS


@dataclasses.dataclass
class D4PGConfig:
    """Configuration options for the D4PG agent."""
    obj_func: str = 'var'
    critic_loss_type: str = 'qr-huber'
    threshold: float = 0.95
    discount: float = 0.99
    batch_size: int = 256
    prefetch_size: int = 4
    target_update_period: int = 100
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: Optional[float] = 32.0
    n_step: int = 20
    sigma: float = 0.3
    clipping: bool = True
    replay_table_name: str = 'priority_table'
    quantile_interval:float =0.01


class ReplayBuffer:
    def __init__(self, obs_shape, action_shape,
                 capacity=100000,
                 batch_size=256,
                 n_step=5,
                 gamma=0.99,
                 device='cpu'):
        """
        Args:
            obs_shape: 观察空间 shape (tuple)
            action_shape: 动作空间 shape (tuple)
            capacity: 最大存储大小
            batch_size: 采样 batch 大小
            n_step: N-step return 步长
            gamma: 折扣因子
        """
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.device = device

        self.buffer = []
        self.n_step_cache = []
        self.lock = threading.Lock()

    def add(self, obs, action, reward, next_obs, done):

        self.n_step_cache.append((obs, action, reward, next_obs, done))

        if len(self.n_step_cache) < self.n_step:
            return

        R = 0.0
        discount = 1.0
        steps_used = 0
        done_at_index = None

        for i, (_, _, r, _, d) in enumerate(self.n_step_cache):
            R += r * discount
            steps_used += 1
            if d:
                done_at_index = i
                break
            discount *= self.gamma

        # γ^steps_used：实际 bootstrapping 折扣
        discount_for_bootstrap = self.gamma ** steps_used

        # 当前 transition 起点
        first_obs, first_action = self.n_step_cache[0][0], self.n_step_cache[0][1]

        # 判断 next_obs / done
        if done_at_index is not None:
            n_step_obs = self.n_step_cache[done_at_index][3]
            last_done = True
        else:
            n_step_obs = self.n_step_cache[-1][3]
            last_done = self.n_step_cache[-1][4]

        with self.lock:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
            self.buffer.append((
                first_obs, first_action,
                np.array(R, dtype=np.float32),
                n_step_obs,
                np.array(discount_for_bootstrap, dtype=np.float32),
                last_done
            ))

        if done_at_index is not None:
            self.n_step_cache.clear()
        else:
            self.n_step_cache.pop(0)

    def sample_batch(self):
        with self.lock:
            batch = random.sample(self.buffer, self.batch_size)

        obs, actions, returns, next_obs, discounts, dones = map(np.array, zip(*batch))
        obs = np.array(obs).reshape(self.batch_size, -1)
        actions = np.array(actions).reshape(self.batch_size, -1)
        next_obs = np.array(next_obs).reshape(self.batch_size, -1)
        return (
            torch.tensor(obs, dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.float32, device=self.device),
            torch.tensor(returns, dtype=torch.float32, device=self.device).squeeze(),
            torch.tensor(next_obs, dtype=torch.float32, device=self.device),
            torch.tensor(discounts, dtype=torch.float32, device=self.device).squeeze(),
            torch.tensor(dones, dtype=torch.float32, device=self.device).squeeze(),
        )

class ReplayDataset(IterableDataset):
    """PyTorch IterableDataset wraps ReplayBuffer to an iterable PyTorch object which can be used in DataLoader"""
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __iter__(self):
        while True:
            if len(self.replay_buffer.buffer) < self.replay_buffer.batch_size:
                time.sleep(0.001)
                continue

            yield self.replay_buffer.sample_batch()

class RunningMeanStd:
    """
    Used as an Observation Normalization class
    """
    def __init__(self, eps=1e-8):
        self.mean = 0
        self.var = 1
        self.count = eps

    def update(self, x):
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = len(x)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class D4PG:
    """D4PG Agent.
    This implements a single-process D4PG agent. This is an actor-critic algorithm
    that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policy (and as a result the
    behavior) by sampling uniformly from this buffer.
    """

    def __init__(
            self,
            policy_network,
            critic_network,
            observation_network=nn.Identity,
            obj_func='var',
            critic_loss_type='c51',
            threshold=0.95,
            quantile_interval: float = 0.01,
            discount: float = 0.99,
            prefetch_size: int = 4,
            target_update_period: int = 100,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            min_replay_size: int = 1000,
            max_replay_size: int = 1000000,
            samples_per_insert: float = 32.0,
            n_step: int = 5,
            sigma: float = 3,
            clipping: bool = True,
            replay_table_name: str = 'priority_table',
            device="cuda",
            log_dir="./logs",
            ckpt_dir="./checkpoints",
    ):
        """Initialize the agent.
        Args:
          policy_network: the online (optimized) policy.
          critic_network: the online critic.
          observation_network: optional network to transform the observations before
            they are fed into any network.
          discount: discount to use for TD updates.
          prefetch_size: size to prefetch from replay.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          min_replay_size: minimum replay size before updating.
          max_replay_size: maximum replay size.
          samples_per_insert: number of samples to take from replay for every insert
            that is made.
          n_step: number of steps to squash into a single transition.
          sigma: standard deviation of zero-mean, Gaussian exploration noise.
          clipping: whether to clip gradients by global norm.
          replay_table_name: string indicating what name to give the replay table.
        """

        self.D4PGConfig = D4PGConfig(
                obj_func=obj_func,
                critic_loss_type=critic_loss_type,
                threshold=threshold,
                discount=discount,
                quantile_interval=quantile_interval,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                min_replay_size=1,  # Let the Agent class handle this.
                max_replay_size=max_replay_size,
                samples_per_insert=None,  # Let the Agent class handle this.
                n_step=n_step,
                sigma=sigma,
                clipping=clipping,
                replay_table_name=replay_table_name,
            )
        self.rms = RunningMeanStd()
        self.device = device
        self.observation_network = observation_network.to(device)
        self.policy_network = policy_network.to(device)
        self.critic_network = critic_network.to(device)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # Target networks are just a copy of the online networks.
        self.target_policy = copy.deepcopy(self.policy_network).to(device)
        self.target_critic = copy.deepcopy(self.critic_network).to(device)
        self.target_observation = copy.deepcopy(self.observation_network).to(device)
        for p in self.target_policy.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False
        # distinguished actor_policy network and policy_network so that avoid in-place modification error
        self.actor_policy = copy.deepcopy(self.policy_network).to(device)
        for p in self.actor_policy.parameters():
            p.requires_grad = False

        self.writer = SummaryWriter(log_dir)
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

    def handle_step(self, step, critic_loss, policy_loss, q_mean):

        # tensorboard
        if self.writer:
            self.writer.add_scalar("loss/critic", critic_loss, step)
            self.writer.add_scalar("loss/policy", policy_loss, step)
            self.writer.add_scalar("q_mean", q_mean, step)

        # save checkpoint
        if step % 200 == 0:
            self.save_checkpoint(step)


    def save_checkpoint(self, step):

        path = f"{self.ckpt_dir}/ckpt_{step}.pth"
        torch.save({
            "observation": self.observation_network.state_dict(),
            "policy": self.policy_network.state_dict(),
            "critic": self.critic_network.state_dict(),
            "t_observation": self.target_observation.state_dict(),
            "t_policy": self.target_policy.state_dict(),
            "t_critic": self.target_critic.state_dict(),
        }, path)

    def save_rms(self, path: str):
        """save RMS for test purpose"""
        rms_state = {
            "mean": self.rms.mean,
            "var": self.rms.var,
            "count": self.rms.count
        }
        torch.save(rms_state, path)

    def load_rms(self, path: str):
        """load RMS"""
        rms_state = torch.load(path,weights_only=False)
        self.rms.mean = rms_state["mean"]
        self.rms.var = rms_state["var"]
        self.rms.count = rms_state["count"]


    def load_checkpoint(self, path,device="cuda"):
        ckpt = torch.load(path, map_location=device)

        self.observation_network.load_state_dict(ckpt["observation"])
        self.policy_network.load_state_dict(ckpt["policy"])
        self.critic_network.load_state_dict(ckpt["critic"])

        self.target_observation.load_state_dict(ckpt["t_observation"])
        self.target_policy.load_state_dict(ckpt["t_policy"])
        self.target_critic.load_state_dict(ckpt["t_critic"])


    def actor_loop(self, env, adder, sync_queue, stop_event):
        obs = env.reset()
        # obs normalize samples
        self.rms.update(obs)
        obs_norm = (obs - self.rms.mean) / self.rms.std
        step = 0
        while not stop_event.is_set():
            obs_tensor = torch.tensor(obs_norm[None, :], dtype=torch.float32, device=self.device)
            obs_tensor = self.observation_network(obs_tensor)
            with torch.no_grad():
                action_tensor = self.actor_policy(obs_tensor)
                action = action_tensor.cpu().numpy()[0]
            action = action + np.random.normal(0, self.D4PGConfig.sigma, size=1)
            action = np.clip(action, 0, 1)
            next_obs, reward, done, _ = env.step(action)
            self.rms.update(next_obs)
            next_obs_norm = (next_obs - self.rms.mean) / self.rms.std
            adder.add(
                obs=obs_norm,
                action=action,
                reward=reward,
                next_obs=next_obs_norm,
                done=done
            )

            if done:
                obs = env.reset()
                self.rms.update(obs)
                obs_norm = (obs - self.rms.mean) / self.rms.std
            else:
                obs = next_obs
                obs_norm = next_obs_norm
            step += 1

            if step % 10 == 0 and not sync_queue.empty():
                new_state_dict = sync_queue.get()
                self.actor_policy.load_state_dict({k: v.to(self.device) for k, v in new_state_dict.items()})


    def learner_loop(self, dataset: IterableDataset, sync_queue, stop_event, num_episodes):
        learner = D4PGLearner(self.policy_network, self.critic_network, self.target_policy, self.target_critic, observation_network = self.observation_network ,
                              target_observation_network = self.target_observation ,discount=self.D4PGConfig.discount,
                               target_update_period = self.D4PGConfig.target_update_period, dataset_iterator = dataset,  obj_func=self.D4PGConfig.obj_func,
                            critic_loss_type=self.D4PGConfig.critic_loss_type, threshold = self.D4PGConfig.threshold, actor_lr = self.actor_lr, critic_lr = self.critic_lr,
                              quantile_interval = self.D4PGConfig.quantile_interval )
        step = 0

        while not stop_event.is_set() and step < num_episodes:

            critic_loss, policy_loss, q_mean = learner.step()

            self.handle_step(step, critic_loss, policy_loss, q_mean)

            if step % 10 == 0:

                self.target_policy.load_state_dict(self.policy_network.state_dict())
                self.target_critic.load_state_dict(self.critic_network.state_dict())
                self.target_observation.load_state_dict(self.observation_network.state_dict())

                sync_queue.queue.clear()
                sync_queue.put({k: v.cpu() for k, v in self.policy_network.state_dict().items()})

            step += 1



class VegaHedgeAgent:
    '''
    This is the Delta-Vega Agent implementation.
    Output: Hedging Actions - Alpha, computed analytically following the alpha approach defined in the paper.
    '''

    def __init__(self, running_env) -> None:
        self.env = running_env


    def select_action(self, observation: np.ndarray) -> np.ndarray:
        episode = self.env.sim_episode
        t = self.env.t
        current_vega = observation[3]
        hedge_option = self.env.portfolio.hed_port.options[episode, t]
        hed_share = -current_vega / hedge_option.vega_path[t] / self.env.portfolio.utils.contract_size
        # action constraints
        gamma_action_bound = float(-self.env.portfolio.get_gamma(t) / \
                             self.env.portfolio.hed_port.options[episode, t].gamma_path[
                                 t] / self.env.portfolio.utils.contract_size)
        action_low = [0, gamma_action_bound]
        action_high = [0, gamma_action_bound]

        if FLAGS.vega_obs:
            # vega bounds
            vega_action_bound = float(-self.env.portfolio.get_vega(t) / \
                                self.env.portfolio.hed_port.options[episode, t].vega_path[
                                    t] / self.env.portfolio.utils.contract_size)
            action_low.append(vega_action_bound)
            action_high.append(vega_action_bound)

        low_val = np.min(action_low)
        high_val = np.max(action_high)

        alpha = (hed_share - low_val) / (high_val - low_val)

        return np.array([alpha])


class GammaHedgeAgent:
    """
    This is the baseline-Delta Gamma Agent implementation.
    Output: Hedging Actions - Alpha, computed analytically following the alpha approach defined in the paper.
    """

    def __init__(self, running_env, hedge_ratio=1.0) -> None:
        self.env = running_env
        self.hedge_ratio = hedge_ratio

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        episode = self.env.sim_episode
        t = self.env.t
        current_gamma = observation[1]

        hedge_gamma = self.hedge_ratio * current_gamma
        hedge_option = self.env.portfolio.hed_port.options[episode, t]
        hed_share = -hedge_gamma / hedge_option.gamma_path[t] / self.env.portfolio.utils.contract_size
        # action constraints
        gamma_action_bound = float(-self.env.portfolio.get_gamma(t) / \
                             self.env.portfolio.hed_port.options[episode, t].gamma_path[
                                 t] / self.env.portfolio.utils.contract_size)
        action_low = [0, gamma_action_bound]
        action_high = [0, gamma_action_bound]

        if FLAGS.vega_obs:
            # vega bounds
            vega_action_bound = float(-self.env.portfolio.get_vega(t) / \
                                self.env.portfolio.hed_port.options[episode, t].vega_path[
                                    t] / self.env.portfolio.utils.contract_size)
            action_low.append(vega_action_bound)
            action_high.append(vega_action_bound)

        low_val = np.min(action_low)
        high_val = np.max(action_high)

        alpha = (hed_share - low_val) / (high_val - low_val)

        return np.array([alpha])


class DeltaHedgeAgent:
    """
    This is the baseline Delta Heging agent implementation
    Output: Hedging Actions - Alpha, computed analytically following the alpha approach defined in the paper.
    """

    def __init__(self, running_env, hedge_ratio=1.0) -> None:
        self.env = running_env
        self.hedge_ratio = hedge_ratio

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        episode = self.env.sim_episode
        t = self.env.t

        hed_share = 0
        # action constraints
        gamma_action_bound = float(-self.env.portfolio.get_gamma(t) / \
                             self.env.portfolio.hed_port.options[episode, t].gamma_path[
                                 t] / self.env.portfolio.utils.contract_size)
        action_low = [0, gamma_action_bound]
        action_high = [0, gamma_action_bound]

        if FLAGS.vega_obs:
            # vega bounds
            vega_action_bound = float(-self.env.portfolio.get_vega(t) / \
                                self.env.portfolio.hed_port.options[episode, t].vega_path[
                                    t] / self.env.portfolio.utils.contract_size)
            action_low.append(vega_action_bound)
            action_high.append(vega_action_bound)

        low_val = np.min(action_low)
        high_val = np.max(action_high)

        alpha = (hed_share - low_val) / (high_val - low_val)

        return np.array([alpha])

