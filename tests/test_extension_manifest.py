import json
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parents[1] / "extension"
MANIFEST = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))


def test_is_manifest_v3_with_service_worker():
    assert MANIFEST["manifest_version"] == 3
    assert MANIFEST["background"]["service_worker"] == "background.js"


def test_every_referenced_file_exists():
    files = [MANIFEST["background"]["service_worker"], MANIFEST["action"]["default_popup"], *MANIFEST["icons"].values()]
    for cs in MANIFEST["content_scripts"]:
        files += cs["js"] + cs["css"]
    for f in files:
        assert (EXT / f).is_file(), f


def test_permissions_are_minimal():
    assert MANIFEST["permissions"] == ["storage"]
    hosts = MANIFEST["host_permissions"]
    assert all(h.startswith(("http://localhost", "http://127.0.0.1")) for h in hosts)
    assert MANIFEST["optional_host_permissions"] == ["https://*/*"]


def test_content_script_only_runs_on_youtube():
    assert [cs["matches"] for cs in MANIFEST["content_scripts"]] == [["https://www.youtube.com/*"]]


@pytest.mark.parametrize("name", ["content.js", "background.js", "popup.js"])
def test_scripts_never_write_untrusted_text_as_html(name):
    code = (EXT / name).read_text(encoding="utf-8")
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in code, f"{name} usa {forbidden}"


def test_background_rejects_non_https_remote_api_urls():
    code = (EXT / "background.js").read_text(encoding="utf-8")
    assert 'u.protocol === "https:"' in code and "localhost" in code and "sender.id !== chrome.runtime.id" in code
