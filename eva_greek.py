from Utils import Utils
from tqdm import tqdm
import tensorflow as tf
import csv
from environment.Environment import TradingEnv
from agent.agent_torch import VegaHedgeAgent, DeltaHedgeAgent, GammaHedgeAgent
import numpy as np
import os
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

def compute_var_cvar(pnl, alpha=0.05):

    pnl_sorted = np.sort(pnl)
    idx = int(alpha * len(pnl_sorted))
    idx = max(1, min(idx, len(pnl_sorted) - 1))
    std = np.std(pnl_sorted)
    mean = np.mean(pnl_sorted)
    var = pnl_sorted[idx]
    cvar = pnl_sorted[:idx].mean()
    return mean, std, var, cvar

def evaluate_policy(agent, env, model_name = None, num_episodes=200):
    if env.logger and model_name is not None:
        env.logger.model_name = model_name
    rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
            action = agent.select_action(obs_tf)  # deterministic policy
            action = np.clip(action, 0, 1)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.array(rewards)



def main(argv):

    eval_utils = Utils(init_ttm=FLAGS.init_ttm, np_seed=FLAGS.eval_seed, num_sim=FLAGS.eval_sim, spread=FLAGS.spread,
                                              volvol=FLAGS.vov, sabr=FLAGS.sabr, gbm=FLAGS.gbm, hed_ttm=FLAGS.hed_ttm,
                                              init_vol=FLAGS.init_vol, poisson_rate=FLAGS.poisson_rate,
                                              moneyness_mean=FLAGS.moneyness_mean, moneyness_std=FLAGS.moneyness_std,
                                              mu=0.0, ttms=[int(ttm) for ttm in FLAGS.liab_ttms],
                                              action_low=float(FLAGS.action_space[0]), action_high=float(FLAGS.action_space[1]))
    # environment = TradingEnv(utils=eval_utils)
    path = "New_folder"

    log_path = os.path.join("logs",path, "logs","eval_log.csv")
    environment = TradingEnv(utils=eval_utils, logger=Logger(log_path))

    if environment.portfolio.crash_day is not None:
        print(f"[Stress Test] Crash injected at day {environment.portfolio.crash_day}")

    agent_gamma = GammaHedgeAgent(environment)
    agent_delta = DeltaHedgeAgent(environment)

    agent = [agent_delta,agent_gamma]
    if FLAGS.vega_obs:
        agent_vega = VegaHedgeAgent(environment)
        agent.append(agent_vega)

    greek_result= []
    mean = []
    std = []
    var = []
    cvar = []

    print("\n🚀 begin model validating...")
    for agent in agent :

        result = evaluate_policy(agent, environment,num_episodes=5000, model_name = agent)
        mean_agent, std_agent, var_agent, cvar_agent = compute_var_cvar(result,alpha=0.05)
        mean.append(mean_agent)
        std.append(std_agent)
        var.append(var_agent)
        cvar.append(cvar_agent)
        greek_result.append(result)

    print('mean','std','Var','CVar')
    print('delta hedging', mean[0],std[0],var[0],cvar[0])
    print('gamma hedging',mean[1],std[1],var[1],cvar[1])

    if FLAGS.vega_obs:
        print('vega hedging', mean[2], std[2],var[2], cvar[2])


if __name__ == '__main__':
    app.run(main)