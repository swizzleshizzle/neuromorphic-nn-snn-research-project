"""EXP-078 seam: resume a cell from a checkpoint. Each docstring names the bug it catches."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuromorphic.envs.cube import N_ACTIONS
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import lookahead as la
from neuromorphic.training.cube_baseline import CubeConfig, make_agent, shell_states


def _agent_and_head(seed=0):
    agent = make_agent(CubeConfig(arm="regionalized", seed=seed))
    torch.manual_seed(seed)
    head = nn.Linear(agent.content, N_ACTIONS)
    return agent, head


def _shell(depth, n):
    return shell_states(ExactBFSDistance(max_depth=depth), depth)[:n]


class _HashCritic:
    reads_states = True

    def __call__(self, states, generator):
        return torch.tensor([float(hash(s) % 1_000_003) for s in states])


STATES = _shell(5, 7)


def _run(agent, head, mode="E", critic=None, **kw):
    # Mode E follows the policy's logits, which are drawn from the evaluation generator, so
    # a resume that loses the generator state changes the outcome. Mode C ignores the logits
    # and could not see it.
    return la.evaluate_lookahead(agent, head, STATES, depth=5, mode=mode, k=2,
                                 generator=torch.Generator().manual_seed(11), rng_seed=11,
                                 imag_seed=11, critic=critic, no_revisit=True, **kw)


OUTCOME = ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
           "goal_fired_frac", "first_fire_len", "fallback_frac", "greedy_modal_action_frac")


def test_default_path_has_no_new_keys():
    """Catches the resume seam leaking into every record: without a checkpoint the result has
    no resumes or elapsed_s key."""
    agent, head = _agent_and_head()
    r = _run(agent, head)
    assert "resumes" not in r and "elapsed_s" not in r


@pytest.mark.parametrize("m", [3, 4])
def test_killed_and_resumed_equals_straight_through(tmp_path, m):
    """THE RESUME TEST. Catches a resume that does not restore the evaluation generator, a
    counter, or the state index: a cell stopped after m states (3 is not a multiple of the
    interval 2, 4 is) and resumed equals the same cell run straight through in every outcome
    field. Uses the real agent, so hidden agent state between episodes would also show."""
    agent, head = _agent_and_head()
    ref = _run(agent, head)
    ck = tmp_path / "cell.ckpt"
    agent, head = _agent_and_head()
    assert _run(agent, head, checkpoint=ck, checkpoint_every=2, stop_after=m) is None
    assert ck.exists()
    agent, head = _agent_and_head()
    got = _run(agent, head, checkpoint=ck, checkpoint_every=2)
    for f in OUTCOME:
        assert got[f] == ref[f], f
    assert got["resumes"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_must_restore_the_generator(tmp_path):
    """Catches the resume test being blind to the generator: with the saved generator state
    replaced by another stream's, the resumed cell must differ from the straight run."""
    agent, head = _agent_and_head()
    ref = _run(agent, head)
    ck = tmp_path / "cell.ckpt"
    agent, head = _agent_and_head()
    _run(agent, head, checkpoint=ck, checkpoint_every=2, stop_after=3)
    saved = torch.load(ck, weights_only=False)
    saved["gen"] = torch.Generator().manual_seed(999).get_state()
    torch.save(saved, ck)
    agent, head = _agent_and_head()
    got = _run(agent, head, checkpoint=ck, checkpoint_every=2)
    assert any(got[f] != ref[f] for f in OUTCOME)


def test_a_finished_checkpoint_returns_without_re_evaluating(tmp_path):
    """Catches a crash after the final checkpoint forcing a full re-run: a second call on a
    finished checkpoint returns the same result and evaluates no state (the critic is never
    called)."""
    agent, head = _agent_and_head()
    ck = tmp_path / "cell.ckpt"
    first = _run(agent, head, mode="C", critic=_HashCritic(), checkpoint=ck, checkpoint_every=2)

    class _Boom(_HashCritic):
        def __call__(self, states, generator):
            raise AssertionError("re-evaluated a finished cell")

    again = la.evaluate_lookahead(agent, head, STATES, depth=5, mode="C", k=2,
                                  generator=torch.Generator().manual_seed(11), rng_seed=11,
                                  imag_seed=11, critic=_Boom(), no_revisit=True,
                                  checkpoint=ck, checkpoint_every=2)
    for f in OUTCOME:
        assert again[f] == first[f], f


def test_a_checkpoint_from_another_cell_is_refused(tmp_path):
    """Catches resuming the wrong cell: a checkpoint written for k 2 is refused by a k 3 call,
    and one written for 7 states by a 6-state call."""
    agent, head = _agent_and_head()
    ck = tmp_path / "cell.ckpt"
    _run(agent, head, checkpoint=ck, checkpoint_every=2, stop_after=2)
    with pytest.raises(ValueError, match=r"checkpoint k is"):
        la.evaluate_lookahead(agent, head, STATES, depth=5, mode="E", k=3,
                              generator=torch.Generator().manual_seed(11), rng_seed=11,
                              imag_seed=11, critic=_HashCritic(), no_revisit=True,
                              checkpoint=ck)
    with pytest.raises(ValueError, match="n_states"):
        la.evaluate_lookahead(agent, head, STATES[:6], depth=5, mode="E", k=2,
                              generator=torch.Generator().manual_seed(11), rng_seed=11,
                              imag_seed=11, critic=_HashCritic(), no_revisit=True,
                              checkpoint=ck)


def test_trace_with_checkpoint_is_refused(tmp_path):
    """Catches a resumed trace silently missing its first episodes."""
    agent, head = _agent_and_head()
    with pytest.raises(ValueError):
        _run(agent, head, checkpoint=tmp_path / "c.ckpt", trace=[])


class _SpyCritic:
    """Records the first value drawn from the imagination generator on every call."""
    reads_states = True

    def __init__(self):
        self.log = []

    def __call__(self, states, generator):
        v = torch.rand(len(states), generator=generator)
        self.log.append(float(v[0]))
        return v


def _spy_run(spy, **kw):
    agent, head = _agent_and_head()
    return la.evaluate_lookahead(agent, head, STATES, depth=5, mode="C", k=1, critic=spy,
                                 generator=torch.Generator().manual_seed(11), rng_seed=11,
                                 imag_seed=11, no_revisit=True, **kw)


@pytest.mark.parametrize("m", [3, 4])
def test_resume_replays_the_same_imagination_seeds(tmp_path, m):
    """Catches a wrong imagination-seed state index after a resume (e.g. i - start instead of
    i): the critic's imagination stream, seen through a spy, in the stopped run followed by the
    resumed run must equal the straight-through run's, call for call."""
    ref = _SpyCritic()
    _spy_run(ref)
    ck = tmp_path / "c.ckpt"
    first, second = _SpyCritic(), _SpyCritic()
    assert _spy_run(first, checkpoint=ck, checkpoint_every=2, stop_after=m) is None
    _spy_run(second, checkpoint=ck, checkpoint_every=2)
    assert len(ref.log) > 0 and len(second.log) > 0
    assert first.log + second.log == ref.log


def test_a_transient_permission_error_on_save_is_retried(tmp_path, monkeypatch):
    """Catches one Defender PermissionError on os.replace crashing a cell: the first replace
    raises, the retry succeeds, the run completes and the checkpoint exists."""
    real = la.os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("Defender")
        return real(src, dst)

    monkeypatch.setattr(la.os, "replace", flaky)
    monkeypatch.setattr(la.time, "sleep", lambda s: None)
    agent, head = _agent_and_head()
    ck = tmp_path / "cell.ckpt"
    r = _run(agent, head, checkpoint=ck, checkpoint_every=2)
    assert r is not None and ck.exists() and calls["n"] > 1


def test_a_persistent_save_failure_is_reraised_after_three_attempts(tmp_path, monkeypatch):
    """Catches the retry swallowing a real fault: three failed attempts re-raise."""
    calls = {"n": 0}

    def broken(src, dst):
        calls["n"] += 1
        raise PermissionError("locked")

    monkeypatch.setattr(la.os, "replace", broken)
    monkeypatch.setattr(la.time, "sleep", lambda s: None)
    agent, head = _agent_and_head()
    with pytest.raises(PermissionError):
        _run(agent, head, checkpoint=tmp_path / "cell.ckpt", checkpoint_every=2)
    assert calls["n"] == 3
