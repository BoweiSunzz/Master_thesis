"""D4PG learner implementation."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset
import os
import time
from typing import Dict, Iterator, List, Optional
import numpy as np
from agent.distributional_torch import QuantileLoss
import agent.distributional_torch as ad





class D4PGLearner:
    """D4PG learner.
    This is the learning component of a D4PG agent. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
            self,
            policy_network,
            critic_network,
            target_policy_network,
            target_critic_network,
            discount: float,
            target_update_period: int,
            dataset_iterator: IterableDataset,
            observation_network=nn.Identity(),
            target_observation_network=nn.Identity(),
            quantile_interval: float = 0.01,
            obj_func='var',
            critic_loss_type='c51',
            threshold=0.95,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            clipping: bool = True,
    ):
        """Initializes the learner.
        Args:
          policy_network: the online (optimized) policy.
          critic_network: the online critic.
          target_policy_network: the target policy (which lags behind the online
            policy).
          target_critic_network: the target critic.
          discount: discount to use for TD updates.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          obj_func: objective function for policy gradient update. (var or cvar)
          critic_loss_type: c51 or qr.
          threshold: threshold for objective function
          dataset_iterator: dataset to learn from, whether fixed or from a replay
            buffer (see `acme.datasets.reverb.make_dataset` documentation).
          observation_network: an optional online network to process observations
            before the policy and the critic.
          target_observation_network: the target observation network.
          clipping: whether to clip gradients by global norm.
        """

        # Store online and target networks.
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network
        self._observation_network = observation_network
        self._target_observation_network = target_observation_network

        self._th = threshold
        self._obj_func = obj_func

        self._critic_loss_func = QuantileLoss(loss_type=critic_loss_type.split('-')[1], b_decay=0.9)
        self._critic_type = critic_loss_type
        self.quantile_interval = quantile_interval


        # Other learner parameters.
        self._discount = discount
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self.num_steps = 0
        self._target_update_period = target_update_period

        # Batch dataset and create iterator.
        self._iterator = dataset_iterator

        # Create optimizers if they aren't given.
        self.policy_opt =  optim.Adam(self._policy_network.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self._critic_network.parameters(), lr=critic_lr)



    def _step(self,critic_updates: int = 7) -> tuple[float, float, float]:
        # Update target network
        if self.num_steps % self._target_update_period == 0:
            self._target_policy_network.load_state_dict(self._policy_network.state_dict())
            self._target_critic_network.load_state_dict(self._critic_network.state_dict())
            self._target_observation_network.load_state_dict(self._observation_network.state_dict())

        self.num_steps += 1

        obs, action, reward, next_obs, discount, done = next(self._iterator)

        obs = obs.cuda()
        action = action.cuda()
        reward = reward.cuda()
        next_obs = next_obs.cuda()
        discount = discount.cuda()
        done = done.cuda()

        # Cast the additional discount to match the environment discount dtype.
        discount = discount * (1.0 - done.float())
        for _ in range(critic_updates):
            with torch.no_grad():
                o_t = self._target_observation_network(next_obs)
                o_t = o_t.detach()

            o_tm1 = self._observation_network(obs).detach()

            # critic outputs  QuantileDistribution object
            q_tm1 = self._critic_network(o_tm1, action)

            with torch.no_grad():
                next_action = self._target_policy_network(o_t)
                q_t = self._target_critic_network(o_t, next_action)

            # Critic loss.

            critic_loss = self._critic_loss_func(q_tm1, reward, discount, q_t)
            # TODO
            prev_policy_sd = {k: v.detach().cpu().clone() for k, v in self._policy_network.state_dict().items()}

            self.critic_opt.zero_grad()
            critic_loss.backward()

            if self._clipping:
                torch.nn.utils.clip_grad_norm_(self._critic_network.parameters(), 40)
            # TODO
            critic_grad_norm = 0.0
            for p in self._critic_network.parameters():
                if p.grad is not None:
                    critic_grad_norm += float(p.grad.detach().norm().item())

            self.critic_opt.step()

        # Actor learning.
        for p in self._critic_network.parameters():
            p.requires_grad_(False)

        o_tm1 = self._observation_network(obs)
        dpg_a_t = self._policy_network(o_tm1)

        if self.num_steps % 50 == 0:
            print("actor mean:", f"[Step {self.num_steps:6d}] ",dpg_a_t.mean().item(), "actor min:", dpg_a_t.min().item(),"actor max:", dpg_a_t.max().item())

        dpg_z_t = self._critic_network(o_tm1, dpg_a_t)

        if self._obj_func == 'meanstd':
            dpg_q_t = dpg_z_t.meanstd()
        elif self._obj_func == 'var':
            dpg_q_t = dpg_z_t.var(self._th)
        elif self._obj_func == 'meanvar':
            dpg_q_t = dpg_z_t.meanvar(self._th)
        elif self._obj_func == 'weightedmean':
            dpg_q_t = dpg_z_t.weighted_tail_loss()
        elif self._obj_func == 'cvar':
            dpg_q_t = dpg_z_t.cvar(self._th)
        else:
            dpg_q_t = dpg_z_t.values.mean(dim=-1)

        # TODO Compute dQ/da gradient
        dQ_da = torch.autograd.grad(dpg_q_t.sum(), dpg_a_t, retain_graph=True, create_graph=False)[0]

        if self.num_steps % 50 == 0:
            # print(f"[Debug] dQ/da norm: {dQ_da.norm().item():.4e}")
            print("dQ/da min/max/mean:", dQ_da.min().item(), dQ_da.max().item(), dQ_da.mean().item())

        policy_loss = -dpg_q_t.mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        #TODO
        policy_grad_norm = 0.0
        for p in self._policy_network.parameters():
            if p.grad is not None:
                policy_grad_norm += float(p.grad.detach().norm().item())

        if self._clipping:
            torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), 40)

        self.policy_opt.step()

        for p in self._critic_network.parameters():
            p.requires_grad_(True)

        #TODO
        post_policy_sd = {k: v.detach().cpu().clone() for k, v in self._policy_network.state_dict().items()}
        param_change = sum((post_policy_sd[k] - prev_policy_sd[k]).abs().sum().item() for k in prev_policy_sd)

        if self.num_steps % 50 == 0:
            print(
                f"[diag step {self.num_steps}] critic_loss={critic_loss.item():.4e} policy_loss={policy_loss.item():.4e} "
                f"critic_grad_norm={critic_grad_norm:.4e} policy_grad_norm={policy_grad_norm:.4e} param_change={param_change:.4e}")

        #Losses and Q to track.
        return critic_loss.item(), policy_loss.item(), dpg_z_t.values.mean().item()


    def step(self):
        # Run the learning step.
        critic_loss, policy_loss, q_mean = self._step()

        return critic_loss, policy_loss, q_mean

