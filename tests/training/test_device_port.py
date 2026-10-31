"""EXP-084: the judge and search path on a torch device, with the CPU path byte-identical.

The golden fixture was produced by the UNMODIFIED code (EXP-084 plan, Task 1 Step 0). Compare,
never regenerate: regenerating from modified code would make these tests unable to fail.
GPU tests skip unless torch.cuda.is_available(); the desktop chain runs them.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neuromorphic.training import value_iteration as vi
from neuromorphic.training.device import setup_device

REPO = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((REPO / "tests" / "fixtures" / "exp084_cpu_golden.json").read_text())
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (ROCm or CUDA)")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


golden = _load("exp084_golden", REPO / "experiments" / "084_gpu_port" / "golden.py")
c75 = _load("exp075_cells", REPO / "experiments" / "075_wide_region" / "cells.py")


@pytest.fixture(scope="module")
def current():
    return golden.compute(REPO)


@pytest.mark.parametrize("key", golden.KEYS)
def test_cpu_matches_golden(current, key):
    """Catches any change to the CPU path: encoding, judge, a training step, a probe, a search."""
    assert current[key] == GOLDEN[key]


def _precision():
    return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.get_float32_matmul_precision())


def test_setup_device_cpu_leaves_flags():
    before = _precision()
    assert setup_device("cpu") == torch.device("cpu")
    assert _precision() == before


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU refusal")
def test_setup_device_cuda_unavailable_exits():
    with pytest.raises(SystemExit):
        setup_device("cuda")


def _judge():
    return c75.build_judge(0, "X", c75.load_pretrained("X", 0))


def _states(n=40):
    import random

    from neuromorphic.envs.cube import N_ACTIONS
    return vi.random_walk_states(n, 14, random.Random(1), N_ACTIONS, set())


@gpu
def test_setup_device_cuda_sets_full_fp32():
    setup_device("cuda")
    assert _precision() == (False, False, "highest")


@gpu
def test_judge_encode_exact_stream_on_gpu():
    setup_device("cuda")
    cpu_j = _judge().eval()
    gpu_j = _judge().eval().to("cuda")
    states = _states()
    g1, g2 = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
    s1, s2 = cpu_j.encode(states, g1), gpu_j.encode(states, g2)
    assert s2.device.type == "cuda"
    assert torch.equal(s1, s2.cpu())
    assert torch.equal(g1.get_state(), g2.get_state())
    with torch.no_grad():
        v1 = cpu_j(states, torch.Generator().manual_seed(12))
        v2 = gpu_j(states, torch.Generator().manual_seed(12))
    assert float((v1 - v2.cpu()).abs().max()) < 1e-4


@gpu
def test_judge_encode_device_stream_on_gpu():
    gpu_j = _judge().eval().to("cuda")
    spikes = gpu_j.encode(_states(), torch.Generator(device="cuda").manual_seed(11))
    assert spikes.device.type == "cuda"


def _train(ckpt, n_updates, device="cpu"):
    from neuromorphic.envs.cube import N_ACTIONS
    probe = [(s, 1 + i % 11) for i, s in enumerate(_states(10))]
    return vi.train_judge(_judge(), "W", n_updates=n_updates, batch=20, max_len=14,
                          n_actions=N_ACTIONS, exclude=set(), probe=probe, seed=4,
                          sync_every=100, probe_every=1, draws=1, ckpt_dir=ckpt,
                          log=lambda *_: None, device=device)


def test_cpu_state_json_keys_unchanged(tmp_path):
    """EXP-083's Gate V compares replayed random state to committed state.json files: a CPU run
    must write exactly the keys it always wrote."""
    _train(tmp_path, 2)
    state = json.loads((tmp_path / "state.json").read_text())
    assert set(state) == {"update", "random_state", "generator_state", "probes"}


def test_resume_refuses_device_mismatch(tmp_path):
    _train(tmp_path, 1)
    with pytest.raises(ValueError, match="device"):
        _train(tmp_path, 2, device="cuda")
    state = json.loads((tmp_path / "state.json").read_text())
    state["generator_device"] = "cuda"
    (tmp_path / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="device"):
        _train(tmp_path, 2, device="cpu")


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _tensors(v)


@gpu
def test_device_checkpoint_tensors_on_cpu(tmp_path):
    setup_device("cuda")
    _train(tmp_path, 2, device="cuda")
    for name in ("judge.pt", "jt.pt", "opt.pt"):
        ts = list(_tensors(torch.load(tmp_path / name, weights_only=False)))
        assert ts and all(t.device.type == "cpu" for t in ts), name
    assert json.loads((tmp_path / "state.json").read_text())["generator_device"] == "cuda"


@gpu
def test_device_resume_continues(tmp_path):
    setup_device("cuda")
    a, b = tmp_path / "a", tmp_path / "b"
    straight = _train(a, 4, device="cuda")
    _train(b, 2, device="cuda")
    resumed = _train(b, 4, device="cuda")
    assert abs(straight["loss_last"] - resumed["loss_last"]) < 1e-5
    ra = json.loads((a / "state.json").read_text())["random_state"]
    rb = json.loads((b / "state.json").read_text())["random_state"]
    assert ra == rb


e75 = _load("exp075_evaluate", REPO / "experiments" / "075_wide_region" / "evaluate.py")


def test_load_judge_device_kwarg_cpu():
    judge = e75.load_judge("X", 0, None, 4000, device="cpu")
    assert all(p.device.type == "cpu" for p in judge.parameters())


def test_judge_critic_output_on_cpu():
    critic = e75.JudgeCritic(e75.load_judge("X", 0, None, 4000, device="cpu"))
    out = critic(_states(6), generator=torch.Generator().manual_seed(1))
    assert out.device.type == "cpu" and out.shape == (6,)


@gpu
def test_search_with_gpu_judge():
    """The search's no-revisit mask is a CPU tensor: a device judge must hand back CPU scores."""
    setup_device("cuda")
    e83 = _load("exp083_evaluate", REPO / "experiments" / "083_unseen_depth9" / "evaluate.py")
    judge = e75.load_judge("X", 0, e83.cells.E82_OUT, 16000, device="cuda")
    states = e83.load_states(0, 9, e83.cells.OUT)[:2]
    rec = e83._search(judge, 9, 0, states)
    assert rec["n"] == 2
