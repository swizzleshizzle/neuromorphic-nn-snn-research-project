from neuromorphic.brain import Brain
from neuromorphic.monitor.schema import REGION_OUTPUT_KEY, SCHEMA_VERSION, build_header, render_for_n
from neuromorphic.monitor.tasks import CubeAdapter, GridworldAdapter


def test_schema_version_is_string():
    assert SCHEMA_VERSION == "1.1"


def test_region_output_key_map():
    assert REGION_OUTPUT_KEY == {
        "sensory": "concept",
        "hippocampus": "population",
        "prefrontal": "utility",
        "router": "gate",
        "motor": "action",
    }


def test_render_for_n_ladder():
    assert render_for_n(64) == "dots"
    assert render_for_n(2000) == "dots"
    assert render_for_n(2001) == "cloud"
    assert render_for_n(100_000) == "cloud"
    assert render_for_n(100_001) == "density"


def test_build_header_topology():
    brain = Brain(grid_n=5, seed=0)
    h = build_header(brain, seed=0, adapter=GridworldAdapter(brain.grid_n))

    assert h["schema_version"] == "1.1"
    assert h["brain"]["T"] == brain.T
    assert h["brain"]["seed"] == 0
    assert isinstance(h["brain"]["config_hash"], str) and len(h["brain"]["config_hash"]) == 8

    assert h["task"]["type"] == "gridworld"
    assert h["task"]["grid_n"] == 5
    assert h["task"]["action_labels"] == ["up", "right", "down", "left"]

    ids = [r["id"] for r in h["regions"]]
    assert ids == ["sensory", "hippocampus", "prefrontal", "router", "motor"]

    by_id = {r["id"]: r for r in h["regions"]}
    assert by_id["sensory"]["n_neurons"] == brain.content
    assert by_id["hippocampus"]["n_neurons"] == 150
    assert by_id["prefrontal"]["n_neurons"] == brain.n_actions
    assert by_id["sensory"]["render"] == "dots"

    pathway_ids = [p["id"] for p in h["pathways"]]
    assert pathway_ids == ["sens_hippo", "sens_pfc", "hippo_pfc", "pfc_motor"]


def test_header_is_json_serializable():
    import json

    brain = Brain(grid_n=5, seed=0)
    h = build_header(brain, seed=0, adapter=GridworldAdapter(brain.grid_n))
    json.dumps(h)  # must not raise


def test_build_header_includes_policy_regions():
    from neuromorphic.brain import Brain
    from neuromorphic.monitor.schema import build_header

    brain = Brain(grid_n=5, seed=0)
    h = build_header(brain, seed=0, adapter=GridworldAdapter(brain.grid_n), policy_regions=["sensory"])
    assert h["policy_regions"] == ["sensory"]
    # default is an empty list (backward compatible)
    h2 = build_header(brain, seed=0, adapter=GridworldAdapter(brain.grid_n))
    assert h2["policy_regions"] == []


def test_config_hash_differs_by_task_type():
    # Same brain, same seed: only the adapter (and therefore task type) differs.
    # If _config_hash ever drops "task" from its payload, a cube run and a
    # gridworld run on this brain would collide on run identity.
    brain = Brain(grid_n=5, seed=0)
    h_grid = build_header(brain, seed=0, adapter=GridworldAdapter(brain.grid_n))
    h_cube = build_header(brain, seed=0, adapter=CubeAdapter())
    assert h_grid["brain"]["config_hash"] != h_cube["brain"]["config_hash"]
