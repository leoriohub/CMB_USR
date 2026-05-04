import importlib.util


def test_chi2_script_missing():
    assert importlib.util.find_spec("scripts.chi2_analysis") is None
