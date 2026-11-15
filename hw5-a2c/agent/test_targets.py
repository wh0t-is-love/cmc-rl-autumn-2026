"""Hand-computed unit tests for ComputeValueTargets and MergeTimeBatch."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solution import ComputeValueTargets, MergeTimeBatch, Policy  # noqa: E402


class DummyModel:
    """Ignores observations, returns fixed values."""

    def __init__(self, values):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self._param = torch.nn.Parameter(torch.zeros(1))

    def __call__(self, obs):
        logits = torch.zeros((obs.shape[0], 6))
        return logits, self.values.clone()

    def parameters(self):
        return iter([self._param])


class DummyPolicy(Policy):
    def __init__(self, values):
        self.model = DummyModel(values)


def make_traj(rewards, resets):
    # rewards, resets: lists per step, each of len N
    T, N = len(rewards), len(rewards[0])
    return {
        "rewards": [np.asarray(r, dtype=np.float32) for r in rewards],
        "resets": [np.asarray(r, dtype=bool) for r in resets],
        "state": {"latest_observation": np.zeros((N, 4, 2, 2), dtype=np.float32)},
    }


def targets_of(rewards, resets, last_values, gamma=0.5):
    pol = DummyPolicy(last_values)
    t = make_traj(rewards, resets)
    ComputeValueTargets(pol, gamma=gamma)(t)
    return t["value_targets"]


def close(a, b):
    return np.allclose(a, b, atol=1e-6)


fails = 0

# Case 1: no resets, gamma=0.5, last value 4
# v2 = 3 + .5*4 = 5;  v1 = 2 + .5*5 = 4.5;  v0 = 1 + .5*4.5 = 3.25
got = targets_of([[1], [2], [3]], [[False], [False], [False]], [4.0])
exp = np.array([[3.25], [4.5], [5.0]], dtype=np.float32)
ok = close(got, exp)
fails += not ok
print("case 1 (no resets):", got.ravel(), "expected", exp.ravel(), "OK" if ok else "FAIL")

# Case 2: reset in the middle (t=1) cuts the future from t=1 on
# v2 = 5;  v1 = 2 (no future);  v0 = 1 + .5*2 = 2
got = targets_of([[1], [2], [3]], [[False], [True], [False]], [4.0])
exp = np.array([[2.0], [2.0], [5.0]], dtype=np.float32)
ok = close(got, exp)
fails += not ok
print("case 2 (mid reset): ", got.ravel(), "expected", exp.ravel(), "OK" if ok else "FAIL")

# Case 3: reset on the last step -> no bootstrap at all
# v2 = 3;  v1 = 2 + .5*3 = 3.5;  v0 = 1 + .5*3.5 = 2.75
got = targets_of([[1], [2], [3]], [[False], [False], [True]], [4.0])
exp = np.array([[2.75], [3.5], [3.0]], dtype=np.float32)
ok = close(got, exp)
fails += not ok
print("case 3 (end reset): ", got.ravel(), "expected", exp.ravel(), "OK" if ok else "FAIL")

# Case 4: two envs at once, mixed resets
# env0: like case 1 -> [3.25, 4.5, 5]
# env1: r=[0,1,0], resets=[F,F,T], v=8 -> [0.5, 1, 0]
got = targets_of(
    [[1, 0], [2, 1], [3, 0]],
    [[False, False], [False, False], [False, True]],
    [4.0, 8.0],
)
exp = np.array([[3.25, 0.5], [4.5, 1.0], [5.0, 0.0]], dtype=np.float32)
ok = close(got, exp)
fails += not ok
print("case 4 (batch mix): ", got.tolist(), "expected", exp.tolist(), "OK" if ok else "FAIL")

# ---- MergeTimeBatch alignment: index (t, n) must map to t*N+n in every key
T, N, A = 3, 2, 4
traj = {
    "observations": [
        np.array([[10 * t + n] for n in range(N)], dtype=np.float32) for t in range(T)
    ],
    "actions": [np.array([100 * t + n for n in range(N)]) for t in range(T)],
    "logits": [
        torch.stack([torch.arange(A).float() + 1000 * t + n for n in range(N)])
        for t in range(T)
    ],
    "value_targets": [np.array([0.1 * t + 0.01 * n for n in range(N)], np.float32) for t in range(T)],
    "state": {"env_steps": T},
}
MergeTimeBatch()(traj)
obs, acts, logits, vt = (
    traj["observations"],
    traj["actions"],
    traj["logits"],
    traj["value_targets"],
)
ok = obs.shape == (T * N, 1) and acts.shape == (T * N,) and logits.shape == (T * N, A) and vt.shape == (T * N,)
fails += not ok
print("shapes:", obs.shape, acts.shape, tuple(logits.shape), vt.shape, "OK" if ok else "FAIL")

align_ok = True
for t in range(T):
    for n in range(N):
        i = t * N + n
        align_ok &= obs[i, 0] == 10 * t + n
        align_ok &= acts[i] == 100 * t + n
        align_ok &= torch.allclose(logits[i], torch.arange(A).float() + 1000 * t + n)
        align_ok &= close(vt[i], [0.1 * t + 0.01 * n])
fails += not align_ok
print("alignment (t,n) -> t*N+n:", "OK" if align_ok else "FAIL")

print("\nRESULT:", "ALL PASSED" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
