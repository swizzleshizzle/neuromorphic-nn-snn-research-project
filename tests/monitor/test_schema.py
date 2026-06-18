from neuromorphic.monitor.schema import REGION_OUTPUT_KEY, SCHEMA_VERSION, render_for_n


def test_schema_version_is_string():
    assert SCHEMA_VERSION == "1.0"


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
