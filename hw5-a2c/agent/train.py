"""A2C training script — the notebook's final cell with a CLI and resume.

Downloads the three course helper files (atari_wrappers, env_batch, runners)
on first run if they are missing, and patches atari_wrappers for
gymnasium >= 1.4 (atari envs live in ale-py, not shimmy).

Two-stage recipe (see README):
    python train.py --nenvs 16 --total-steps 10000000 --out runs/n16s45.jsonl
    python train.py --nenvs 16 --total-steps 15000000 --entropy-coef 0.005 \
        --lr 2.3e-4 --out runs/n16s45.jsonl --resume auto
"""
import argparse
import json
import math
import os
import re
import time
import urllib.request

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

BASE = "https://raw.githubusercontent.com/yandexdataschool/Practical_RL/master/week06_policy_based/"
HELPERS = ["atari_wrappers.py", "env_batch.py", "runners.py"]

for fname in HELPERS:
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        print(f"downloading {fname} ...")
        urllib.request.urlretrieve(BASE + fname, path)

with open(os.path.join(HERE, "atari_wrappers.py")) as f:
    src = f.read()
patched = src
# bare top-level shimmy import (master format): wrap in try/except so atari
# envs also work with gymnasium >= 1.4 (ale-py instead of shimmy)
patched = re.sub(
    r"(?m)^from shimmy\.atari_env import AtariEnv$",
    "try:\n    from shimmy.atari_env import AtariEnv\nexcept ImportError:\n"
    "    import ale_py\n    from ale_py.env import AtariEnv\n"
    "    gym.register_envs(ale_py)",
    patched,
)
# fallback imports old `gym`; use the gymnasium `gym` already imported at the
# top of the file instead
patched = patched.replace(
    "except ImportError:\n    import gym\n    import ale_py",
    "except ImportError:\n    import ale_py",
)
if patched != src:
    with open(os.path.join(HERE, "atari_wrappers.py"), "w") as f:
        f.write(patched)

import sys  # noqa: E402

sys.path.insert(0, HERE)

from atari_wrappers import NumpySummaries, nature_dqn_env  # noqa: E402
from runners import EnvRunner  # noqa: E402
from solution import A2C, ComputeValueTargets, MergeTimeBatch, Model, Policy  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="SpaceInvadersNoFrameskip-v4")
    p.add_argument("--nenvs", type=int, default=8)
    p.add_argument("--nsteps", type=int, default=5)
    p.add_argument("--total-steps", type=int, default=10_000_000)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--out", default="runs/run.jsonl")
    p.add_argument("--resume", default="auto",
                   help="'auto' (resume from the checkpoint next to --out if it "
                        "exists), 'never', or a checkpoint path")
    p.add_argument("--log-every", type=int, default=200, help="updates between jsonl rows")
    p.add_argument("--ckpt-every", type=int, default=500_000, help="env steps between checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ckpt_path = os.path.splitext(out_path)[0] + ".ckpt.pt"

    resume_from = None
    if args.resume == "auto" and os.path.exists(ckpt_path):
        resume_from = ckpt_path
    elif args.resume not in ("auto", "never"):
        resume_from = args.resume

    env = nature_dqn_env(args.env, nenvs=args.nenvs, summaries="Numpy")
    n_actions = env.action_space.spaces[0].n

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    model = Model(n_actions).to(device)
    policy = Policy(model)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-5)
    runner = EnvRunner(
        env=env,
        policy=policy,
        nsteps=args.nsteps,
        transforms=[ComputeValueTargets(policy, gamma=args.gamma), MergeTimeBatch()],
    )
    a2c = A2C(policy, optimizer, entropy_coef=args.entropy_coef)

    start_step, updates_done, t_elapsed = 0, 0, 0.0
    if resume_from:
        ck = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        updates_done = ck["updates"]
        t_elapsed = ck.get("t_elapsed", 0.0)
        runner.step_var = start_step  # continue the global step counter
        # warm LR restart: fresh linear decay from --lr over the remaining steps
        for g in optimizer.param_groups:
            g["lr"] = args.lr
        print(f"resumed from {resume_from}: step={start_step} updates={updates_done}")

    remaining = args.total_steps - start_step
    if remaining <= 0:
        raise SystemExit(f"checkpoint is already at {start_step} steps; "
                         f"--total-steps must be larger to continue")
    total_updates = math.ceil(remaining / (runner.nsteps * runner.nenvs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda upd: 1.0 - upd / total_updates
    )

    with open(out_path, "a") as log:
        t0 = time.time() - t_elapsed
        last_ckpt_step = start_step
        while runner.step_var < args.total_steps:
            trajectory = runner.get_next()
            metrics = a2c.step(trajectory)
            scheduler.step()
            updates_done += 1

            if updates_done % args.log_every == 0:
                rewards = NumpySummaries.get_values("Episodes/reward_mean_100")
                row = {
                    "step": runner.step_var,
                    "reward_mean_100": rewards[-1][1] if rewards else None,
                    "updates": updates_done,
                    **metrics,
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                r = row["reward_mean_100"]
                print(f"step={row['step']:>9} reward100={r} "
                      f"entropy={metrics['entropy']:.3f} vloss={metrics['value_loss']:.4f} "
                      f"R2={metrics['r_squared']:.3f} |g|={metrics['grad_norm']:.2f}",
                      flush=True)

            if runner.step_var - last_ckpt_step >= args.ckpt_every \
                    or runner.step_var >= args.total_steps:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": runner.step_var,
                        "updates": updates_done,
                        "t_elapsed": time.time() - t0,
                        "seed": args.seed,
                    },
                    ckpt_path,
                )
                last_ckpt_step = runner.step_var

    print("done")


if __name__ == "__main__":
    main()
