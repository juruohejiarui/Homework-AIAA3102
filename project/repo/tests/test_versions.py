import json

from pipeline.versions import capture_package_versions, write_package_versions


def test_package_version_capture_and_write(tmp_path) -> None:
    captured = capture_package_versions()
    destination = write_package_versions(tmp_path / "versions.json")
    restored = json.loads(destination.read_text(encoding="utf-8"))

    assert captured["python"]["version"]
    assert set(captured["packages"]) == {
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
    }
    assert restored == captured
