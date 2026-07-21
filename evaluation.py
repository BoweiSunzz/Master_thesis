from typing import Mapping, Sequence
from multiprocessing import Process, Queue
import os
import csv

from torch.cuda import device
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from environment.Environment import TradingEnv
from main import make_quantile_networks
from agent.agent_torch import D4PG
from Utils import Utils
from absl import app
from absl import flags
import main

FLAGS = flags.FLAGS


class Logger:
    def __init__(self, filename,model_name="Unknown"):
        self.filename = filename
        self.model_name = model_name
        self.file_exists = os.path.isfile(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Open file in append mode
        self.file = open(filename, 'a', newline='')
        self.writer = None

    def write(self, record: dict):
        """
        record: a dictionary like dataclasses.asdict(StepResult)
        """
        record["model_name"] = self.model_name

        if self.writer is None:
            # Create CSV writer
            fieldnames = list(record.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)

            # Write header only if file didn't already exist
            if not self.file_exists:
                self.writer.writeheader()

        # Write one row
        self.writer.writerow(record)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def compute_var_cvar(pnl, alpha=0.95):
    alpha = 1 - alpha
    pnl_sorted = np.sort(pnl)
    idx = int(alpha * len(pnl_sorted))
    idx = max(1, min(idx, len(pnl_sorted) - 1))
    var = pnl_sorted[idx]
    cvar = pnl_sorted[:idx].mean()
    return var, cvar


@torch.no_grad()
def evaluate_policy(agent, env, num_episodes=200, device="cpu", model_name = None):

    if env.logger and model_name is not None:
        env.logger.model_name = model_name

    rewards = []
    action_stats = []

    agent.policy_network.eval()
    agent.observation_network.eval()

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs_norm = (obs - agent.rms.mean) / agent.rms.std
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            obs_t = agent.observation_network(obs_t)
            action_t = agent.policy_network(obs_t)
            action = action_t.squeeze(0).cpu().numpy()
            if np.isscalar(action):
                action = np.array([action])
            action = np.clip(action, -1, 1)

            action_stats.append(action)

            obs, reward, done, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return np.array(rewards), np.mean(action_stats)

def evaluate_checkpoints(agent, env, checkpoint_dir,
                         num_eval_episodes=200, alpha=0.95, device = "cpu"):

    ckpt_paths = sorted([
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith("ckpt_") and f.endswith(".pth")
    ])

    results = []

    for path in tqdm(ckpt_paths, desc="Evaluating checkpoints"):

        try:
            agent.load_checkpoint(path, device)
        except Exception as e:
            print(f"⚠️ Could not load checkpoint {path}: {e}")
            continue

        rewards,action_mean = evaluate_policy(agent, env, num_eval_episodes, device, model_name = os.path.basename(path))

        mean_r = rewards.mean()
        std_r = rewards.std()
        var, cvar = compute_var_cvar(rewards, alpha)

        results.append({
            "checkpoint": os.path.basename(path),
            "mean_return": mean_r,
            "std_return": std_r,
            "meanstd" : mean_r - 1.645*std_r,
            "VaR": var,
            "CVaR": cvar,
            "action_mean" : action_mean
        })

    df = pd.DataFrame(results)
    df = df.sort_values("meanstd")
    best_ckpt = df.iloc[-1]

    print("\n🏆 Best model:")
    print(best_ckpt)

    results_path = os.path.join(checkpoint_dir, "evaluation_results.csv")
    df.to_csv(results_path, index=False)
    print(f"✅ Saved evaluation results to {results_path}")

    return df, best_ckpt

def eval(argv):

    eval_utils = Utils(init_ttm=FLAGS.init_ttm, np_seed=FLAGS.eval_seed, num_sim=FLAGS.eval_sim, spread=FLAGS.spread,
                                              volvol=FLAGS.vov, sabr=FLAGS.sabr, gbm=FLAGS.gbm, hed_ttm=FLAGS.hed_ttm,
                                              init_vol=FLAGS.init_vol, poisson_rate=FLAGS.poisson_rate,
                                              moneyness_mean=FLAGS.moneyness_mean, moneyness_std=FLAGS.moneyness_std,
                                              mu=0.0, ttms=[int(ttm) for ttm in FLAGS.liab_ttms],
                                              action_low=float(FLAGS.action_space[0]), action_high=float(FLAGS.action_space[1]), stress_test=FLAGS.stress_test)

    path = "spread=0.02_obj=cvar_critic=qr-huber_20260419-033237"

    log_path = os.path.join("logs",path, "logs","eval_log.csv")
    environment = TradingEnv(utils=eval_utils, logger=Logger(log_path))
    # environment = TradingEnv(utils=eval_utils)

    if environment.portfolio.crash_day is not None:
        print(f"[Stress Test] Crash injected at day {environment.portfolio.crash_day}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_networks = make_quantile_networks(action_space=environment.action_space,
                                                observation_space=environment.observation_space,quantile_interval=FLAGS.quantile_interval)
    agent = D4PG(
        obj_func=FLAGS.obj_func,
        threshold=FLAGS.threshold,
        critic_loss_type=FLAGS.critic,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network=agent_networks['observation'],
        n_step=FLAGS.n_step,
        discount=FLAGS.discount,
        sigma=0,  # greedy policy when eval
        device=device,
    )

    agent.load_rms(os.path.join("logs",path,"checkpoints","rms_state.pth"))

    print("\n🚀 beginning model evaluation...")
    eval_results, best_ckpt = evaluate_checkpoints(
                agent,
                environment,
                checkpoint_dir= os.path.join("logs",path,"checkpoints"),
                num_eval_episodes=FLAGS.eval_sim,
                alpha=FLAGS.threshold,
                device= device)

if __name__ == '__main__':
    app.run(eval)