from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit", reason="Streamlit is not installed in this execution environment")
from streamlit.testing.v1 import AppTest


def test_streamlit_app_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
