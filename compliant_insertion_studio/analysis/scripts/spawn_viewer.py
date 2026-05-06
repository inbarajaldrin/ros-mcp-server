#!/usr/bin/env python3
# Reference: spawn the compare HTML viewer with two episodes pre-loaded.
# Decouples data loading from plot rendering so I (the agent) can iterate on
# what to plot/compare without touching the data exporter.
#
# Generates a self-contained HTML by reading compare.html and injecting:
#   <script>SLOTS.A = {...}; SLOTS.B = {...}; renderMeta('A'); renderMeta('B'); redraw();</script>
# at the end. Result is openable directly via xdg-open without file:// fetch issues.
#
# Usage:
#   python3 spawn_viewer.py <basename_a> <basename_b>
# e.g.
#   python3 spawn_viewer.py insert_u_orange_20260505_193645 insert_u_orange_20260505_193941

import argparse
import json
import os
import subprocess
import sys
import tempfile
sys.path.insert(0, os.path.dirname(__file__))
from _paths import DATA_DIR, ANALYSIS_DIR


def load_episode(basename: str) -> dict:
    """Load the viewer JSON for a basename. Auto-export if missing."""
    json_path = os.path.join(str(DATA_DIR), f"viewer_{basename}.json")
    if not os.path.exists(json_path):
        print(f"  no JSON found, exporting from raw...")
        rc = subprocess.run([sys.executable,
                              os.path.join(os.path.dirname(__file__),
                                           "22_export_episode_for_viewer.py"),
                              basename],
                             check=False).returncode
        if rc != 0 or not os.path.exists(json_path):
            raise RuntimeError(f"export failed for {basename}")
    return json.load(open(json_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("basename_a")
    p.add_argument("basename_b")
    p.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    p.add_argument("--out", default=None, help="output HTML path")
    args = p.parse_args()

    print(f"loading A: {args.basename_a}")
    data_a = load_episode(args.basename_a)
    print(f"loading B: {args.basename_b}")
    data_b = load_episode(args.basename_b)

    # Read the base compare.html
    base_html = os.path.join(str(ANALYSIS_DIR), "viewer", "compare.html")
    with open(base_html) as fh:
        html = fh.read()

    # Inject the data + auto-load just before </body>
    inject = (
        "<script>\n"
        f"window.SLOTS.A = {json.dumps(data_a)};\n"
        f"window.SLOTS.B = {json.dumps(data_b)};\n"
        "renderMeta('A'); renderMeta('B'); redraw();\n"
        "</script>\n"
    )
    if "</body>" in html:
        html = html.replace("</body>", inject + "</body>")
    else:
        html = html + inject

    # Make SLOTS available on window so the inject can write to it
    html = html.replace("const SLOTS = ", "window.SLOTS = ", 1)

    # Write out
    out_path = args.out or os.path.join(
        str(ANALYSIS_DIR), "viewer",
        f"snapshot_{args.basename_a}_vs_{args.basename_b}.html",
    )
    with open(out_path, "w") as fh:
        fh.write(html)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nwrote: {out_path}  ({size_mb:.1f} MB)")

    if not args.no_open:
        subprocess.Popen(["xdg-open", out_path], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
        print(f"  opened in browser")


if __name__ == "__main__":
    main()
