import json
from pathlib import Path


def test_powerloss_notebook_has_no_chi2_import():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "04_PowerLoss_Explorer.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        assert (
            "from scripts.chi2_analysis    import analyse as chi2_analyse" not in source
        )
