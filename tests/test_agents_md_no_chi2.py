from pathlib import Path


def test_agents_md_has_no_chi2_entry():
    agents_md = Path(__file__).resolve().parents[1] / "AGENTS.md"
    content = agents_md.read_text(encoding="utf-8")

    assert "chi2_analysis.py" not in content


def test_agents_md_planck_data_description():
    agents_md = Path(__file__).resolve().parents[1] / "AGENTS.md"
    content = agents_md.read_text(encoding="utf-8")

    assert (
        "planck_data.py        — Planck 2018 low-ℓ TT data loader (Commander, IRSA R3)"
        in content
    )
