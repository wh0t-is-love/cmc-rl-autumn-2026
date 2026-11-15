"""Live evaluation of a saved checkpoint, fully independent of the training loop.

Runs the policy in fresh environments (raw rewards, no clipping) and prints
per-episode scores — both with sampled actions (as during training) and argmax.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/wil/labs/rl_hw")
sys.path.insert(0, "/home/wil/labs/rl_hw/src")

from atari_wrappers import nature_dqn_env  # noqa: E402
from solution import Model  # noqa: E402

CKPT = sys.argv[1]
N_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ARGMAX_EPISODES = int(sys.argv[3]) if len(sys.argv) > 3 else 2
MAX_STEPS = 20000

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("ckpt keys:", list(ck.keys()) if isinstance(ck, dict) else type(ck))
state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

model = Model(6)  # Space Invaders: 6 actions
model.load_state_dict(state)
model.eval()

# single env, NO reward clipping, NO summaries — raw rewards only
env = nature_dqn_env("SpaceInvadersNoFrameskip-v4", summaries=False, clip_reward=False)


def run_episode(argmax):
    obs, _ = env.reset()
    total, steps, lives_total = 0.0, 0, 0
    while steps < MAX_STEPS:
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(obs[None], dtype=torch.float32))
        if argmax:
            a = int(logits.argmax(dim=-1).item())
        else:
            a = int(torch.distributions.Categorical(logits=logits).sample().item())
        obs, r, term, trunc, info = env.step(a)
        total += float(r)
        steps += 1
        if info.get("real_done", False) or (term or trunc):
            if info.get("real_done", False):
                break
            obs, _ = env.reset()
    return total, steps


print(f"\nsampled-policy episodes ({N_EPISODES}):")
for i in range(N_EPISODES):
    total, steps = run_episode(argmax=False)
    print(f"  episode {i + 1}: raw score = {total:.0f}  ({steps} agent-steps)")

print(f"\nargmax-policy episodes ({ARGMAX_EPISODES}):")
for i in range(ARGMAX_EPISODES):
    total, steps = run_episode(argmax=True)
    print(f"  episode {i + 1}: raw score = {total:.0f}  ({steps} agent-steps)")

env.close()
