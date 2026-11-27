"""Render the fixed contributor roster as a GitHub-safe PNG using headless Chrome/Edge.

Inspired by the circular-avatar wall in thinkasany/organize-contributors.
No organization lookup, contribution ranking, token, or automatic Git commit.
Run from any directory: python scripts/render_contributors.py
"""

import argparse
import base64
import html
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT / "docs/contributors.json"
OUTPUT = ROOT / "docs/images/contributors.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, help="Chrome or Edge executable")
    args = parser.parse_args()
    members = json.loads(ROSTER.read_text(encoding="utf-8"))
    members.sort(key=lambda member: (member["login"] != "rain-xhy", member["login"].casefold()))
    if len(members) != 6 or len({m["login"] for m in members}) != 6:
        raise ValueError("Expected exactly six distinct contributors")
    candidates = [str(args.browser)] if args.browser else [
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
        "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        shutil.which("chromium"), shutil.which("google-chrome"), shutil.which("chromium-browser"),
    ]
    browser = next((path for path in candidates if path and Path(path).is_file()), None)
    if browser is None:
        raise FileNotFoundError("Install Chrome/Edge or supply --browser")

    tiles = []
    for member in members:
        request = urllib.request.Request(member["avatar"], headers={"User-Agent": "HuCooked-README"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
            mime = response.headers.get_content_type()
        if mime not in ("image/png", "image/jpeg", "image/webp"):
            raise ValueError(f"Unexpected avatar format for {member['login']}: {mime}")
        uri = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
        login = html.escape(member["login"])
        tiles.append(f'<section><img src="{uri}" alt="{login}"><div>{login}</div></section>')

    # Six members share one row at GitHub's desktop reading width.
    # Rendering at 2x keeps circles and usernames crisp when embedded in GitHub.
    page = '''<!doctype html><html><meta charset="utf-8"><style>
html,body{margin:0;width:900px;height:170px;background:#ffffff;overflow:hidden}
main{display:grid;grid-template-columns:repeat(6,1fr);padding:22px 12px}
section{text-align:center;color:#243d43;font:600 17px/26px "Segoe UI",Arial,sans-serif}
img{display:block;width:86px;height:86px;object-fit:cover;border-radius:50%;margin:0 auto 10px;border:3px solid #e4eceb}
section:first-child img{border-color:#63cdb7}
</style><main>''' + "".join(tiles) + "</main></html>"
    work = Path(tempfile.mkdtemp(prefix="hucooked-contributors-"))
    source = work / "index.html"
    source.write_text(page, encoding="utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staged = work / "contributors.png"
    subprocess.run([
        browser, "--headless", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
        "--force-device-scale-factor=2", "--window-size=900,170",
        "--user-data-dir=" + str(work / "profile"), "--screenshot=" + str(staged),
        source.as_uri(),
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if not staged.is_file():
        raise RuntimeError("Browser did not produce a screenshot")
    OUTPUT.write_bytes(staged.read_bytes())
    print("Order: " + ", ".join(member["login"] for member in members))
    print(f"Written: {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
