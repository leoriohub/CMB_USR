from pathlib import Path


def test_agents_md_has_no_chi2_entry():
    agents_md = Path(__file__).resolve().parents[1] / "AGENTS.md"
    content = agents_md.read_text(encoding="utf-8")
    lowered = content.lower()

    assert "chi2_analysis.py" not in content
    assert "chi2" not in lowered
    assert "chi^2" not in lowered
    assert "chi squared" not in lowered


def test_agents_md_planck_data_description():
    agents_md = Path(__file__).resolve().parents[1] / "AGENTS.md"
    content = agents_md.read_text(encoding="utf-8")

    assert "planck_data.py" in content
    assert "Planck 2018" in content
    assert "Commander" in content
