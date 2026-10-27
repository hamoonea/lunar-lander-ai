"""Train, evaluate, watch, and record a Gymnasium Lunar Lander agent."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

ENV = "LunarLander-v3"


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def train(args):
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(vars(args), indent=2))
    env = make_vec_env(ENV, n_envs=4, seed=args.seed, monitor_dir=str(run / "episodes"))
    evaluation = Monitor(gym.make(ENV))
    evaluation.reset(seed=args.seed + 10000)
    try:
        model = (PPO.load(args.resume, env=env, device="cpu") if args.resume else
                 PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1,
                     n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999,
                     gae_lambda=0.98, ent_coef=0.01))
        model.set_logger(configure(str(run), ["stdout", "csv"]))
        callbacks = [
            EvalCallback(evaluation, best_model_save_path=str(run / "best"),
                         log_path=str(run / "eval"), eval_freq=5000,
                         n_eval_episodes=10, deterministic=True),
            CheckpointCallback(save_freq=25000, save_path=str(run / "checkpoints")),
        ]
        try:
            model.learn(total_timesteps=args.steps, callback=callbacks,
                        reset_num_timesteps=not bool(args.resume))
        except KeyboardInterrupt:
            print("Training interrupted; saving current model.")
        finally:
            model.save(run / "final_model")
        print(f"Saved {run / 'final_model.zip'}")
    finally:
        env.close()
        evaluation.close()


def rollout(args):
    import imageio.v2 as imageio
    mode = "rgb_array" if args.video else ("human" if args.command == "watch" else None)
    env = gym.make(ENV, render_mode=mode)
    writer = None
    rewards = []
    model = None if args.random else PPO.load(args.model, device="cpu")
    try:
        if args.video:
            path = Path(args.video)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(path), fps=env.metadata["render_fps"], macro_block_size=1)
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            env.action_space.seed(args.seed + episode)
            total, steps = 0.0, 0
            if writer:
                writer.append_data(env.render())
            while True:
                action = env.action_space.sample() if model is None else int(model.predict(obs, deterministic=True)[0])
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                steps += 1
                if writer:
                    writer.append_data(env.render())
                if terminated or truncated:
                    break
            rewards.append(total)
            print(f"Episode {episode + 1}: reward={total:.1f}, steps={steps}")
    finally:
        env.close()
        if writer:
            writer.close()
    result = {"policy": "random" if args.random else args.model,
              "seed": args.seed, "episodes": args.episodes,
              "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
              "rewards": rewards}
    print(json.dumps(result, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("train", help="Train PPO and save checkpoints and learning metrics")
    p.add_argument("--steps", type=positive, default=1_000_000)
    p.add_argument("--run", default="runs/lander")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", help="Continue from a trusted local model ZIP into a new run")
    for command in ("evaluate", "watch"):
        p = subs.add_parser(command)
        p.add_argument("--model", default="runs/lander/best/best_model.zip")
        p.add_argument("--random", action="store_true", help="Use random actions as a baseline")
        p.add_argument("--episodes", type=positive, default=10 if command == "evaluate" else 3)
        p.add_argument("--seed", type=int, default=20000)
        p.add_argument("--video", help="Record MP4 instead of opening a window")
        p.add_argument("--output", help="Write evaluation results as JSON")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        train(args)
    else:
        rollout(args)


if __name__ == "__main__":
    main()
