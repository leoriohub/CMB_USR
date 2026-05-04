from pathlib import Path


def _find_agents_md(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        candidate = parent / "AGENTS.md"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("AGENTS.md not found in parent directories")


def test_agents_md_has_no_chi2_entry():
    agents_md = _find_agents_md(Path(__file__).resolve())
    content = agents_md.read_text(encoding="utf-8")
    lowered = content.lower()

    assert "chi2_analysis.py" not in content
    assert "chi^2" not in lowered
    assert "chi squared" not in lowered
    assert "χ²" not in content


def test_agents_md_planck_data_description():
    agents_md = _find_agents_md(Path(__file__).resolve())
    content = agents_md.read_text(encoding="utf-8")

    assert "planck_data.py" in content
    assert "Planck 2018" in content
    assert "Commander" in content
