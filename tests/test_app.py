"""Drive the Streamlit app headlessly: load the synthetic demo, train, and check every tab
renders without an exception."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_app_trains_on_synthetic_demo() -> None:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    assert "Synthetic telecom" in at.sidebar.selectbox[0].value
    train = next(b for b in at.button if b.label == "Train models")
    train.click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Expected savings" in labels and "Churn score" in labels
    assert any(d.label.startswith("Download target list") for d in at.get("download_button"))
