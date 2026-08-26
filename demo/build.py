"""Build the static demo site into _site/.

Copies `frontend/` verbatim, so a change there reaches the demo on the next
build with no manual sync, then layers the demo overlay on top: `demo.css`
after `style.css`, `demo-shim.js` before `app.js` (it must patch
`window.fetch` before app.js's first call), `demo-ui.js` after. See
`demo/overlay/demo-shim.js` for why the ordering matters.

Fails loudly if `frontend/` no longer exposes what the overlay depends on,
instead of silently shipping a broken demo - see `_check_contract`.

Usage:
    python demo/build.py
    python demo/build.py && python -m http.server -d _site 8080
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_FRONTEND = _ROOT / "frontend"
_DEMO = Path(__file__).resolve().parent
_OVERLAY = _DEMO / "overlay"
_SITE = _ROOT / "_site"

# What demo-shim.js and demo-ui.js assume is still true of frontend/. If any
# of these stop holding, the overlay silently breaks instead of erroring, so
# a mismatch here fails the build with the overlay file to go fix.
_STREAM_ENDPOINT = "/api/related-articles/stream"
_REQUIRED_IDS = ["topic", "searchBtn", "sourcesPreset", "language", "discoveryNote"]


def _check_contract(html: str, app_js: str) -> None:
    if _STREAM_ENDPOINT not in app_js:
        sys.exit(
            f"frontend/app.js no longer calls {_STREAM_ENDPOINT!r} - "
            "update demo/overlay/demo-shim.js's STREAM_PATH to match."
        )
    missing = [i for i in _REQUIRED_IDS if f'id="{i}"' not in html]
    if missing:
        sys.exit(
            f"frontend/index.html is missing element id(s) {missing} that "
            "demo/overlay/demo-ui.js depends on - update that file to match "
            "the new markup."
        )


def _rewrite_static_paths(html: str) -> str:
    html = html.replace('href="/static/style.css"', 'href="static/style.css"')
    html = html.replace('src="/static/app.js"', 'src="static/app.js"')
    return html


def _set_demo_title(html: str) -> str:
    return html.replace(
        "<title>EBM Lens - Evidence Discovery</title>",
        "<title>EBM Lens - Demo</title>",
    )


def _inject_overlay(html: str) -> str:
    html = html.replace(
        '<link rel="stylesheet" href="static/style.css">',
        '<link rel="stylesheet" href="static/style.css">\n'
        '<link rel="stylesheet" href="overlay/demo.css">',
    )
    html = html.replace(
        '<script src="static/app.js"></script>',
        '<script src="overlay/demo-shim.js"></script>\n'
        '<script src="static/app.js"></script>\n'
        '<script src="overlay/demo-ui.js"></script>',
    )
    return html


def build() -> None:
    if _SITE.exists():
        shutil.rmtree(_SITE)
    (_SITE / "static").mkdir(parents=True)

    html_src = (_FRONTEND / "index.html").read_text(encoding="utf-8")
    app_js_src = (_FRONTEND / "app.js").read_text(encoding="utf-8")
    _check_contract(html_src, app_js_src)

    shutil.copy2(_FRONTEND / "app.js", _SITE / "static" / "app.js")
    shutil.copy2(_FRONTEND / "style.css", _SITE / "static" / "style.css")

    html = _rewrite_static_paths(html_src)
    html = _set_demo_title(html)
    html = _inject_overlay(html)
    (_SITE / "index.html").write_text(html, encoding="utf-8")

    shutil.copytree(_OVERLAY, _SITE / "overlay")

    fixtures_dir = _DEMO / "fixtures"
    if not (fixtures_dir / "index.json").exists():
        sys.exit(
            "demo/fixtures/index.json is missing - run "
            "demo/scripts/record_demo.py before building."
        )
    shutil.copytree(fixtures_dir, _SITE / "fixtures")

    n_fixtures = len(list((_SITE / "fixtures").glob("*.json"))) - 1  # exclude index.json
    print(f"built {_SITE} ({n_fixtures} fixtures)")


if __name__ == "__main__":
    build()
