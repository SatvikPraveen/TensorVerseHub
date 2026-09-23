"""Structural checks for the curriculum notebooks (execution is out of scope for CI)."""

import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS = sorted(NOTEBOOKS_DIR.rglob("*.ipynb"))
FORBIDDEN = ("tf.contrib.", "tf.random.beta(", "from src.", "import src.")


def test_notebook_count():
    assert len(NOTEBOOKS) >= 27


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_valid(path):
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    assert nb.cells, "notebook has no cells"
    assert nb.cells[0].cell_type == "markdown", "first cell should be a title"
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for token in FORBIDDEN:
        assert token not in code, f"{path.name} uses removed API {token}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_json_has_no_large_outputs(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    for cell in raw["cells"]:
        for output in cell.get("outputs", []):
            for key, value in output.get("data", {}).items():
                if key.startswith("image/"):
                    assert len("".join(value)) < 2_000_000, "embedded image output over 2 MB"
