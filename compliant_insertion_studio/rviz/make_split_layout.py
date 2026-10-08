#!/usr/bin/env python3
"""Fit an .rviz config's window to the current screen as a left/right split.

One half is a dock holding one Image display's panel (right by default); the 3D
view fills the rest. RViz persists dock placement only as an opaque QMainWindow::saveState() blob
("QMainWindow State" in the .rviz file), so we build an equivalent QMainWindow
offscreen with docks carrying the same objectNames RViz gives them, and save its
state. RViz names each dock after its panel ("Displays", "Views", ...) or, for a
display's own panel, after the display ("Camera").

Usage:
    make_split_layout.py TEMPLATE.rviz OUT.rviz [--image-dock NAME] [--split 0.5]
                         [--camera-side right|left]
                         [--width W --height H]   # default: primary screen size

Panels listed in the template (Displays, Views, ...) are not placed here: the demo
template lists none, so the window is just robot | camera.
"""
import argparse
import sys

import yaml

def build_state(width, height, image_dock, split, side):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget

    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = QMainWindow()
    # Lay out without mapping: when the screen size was read first, the app is on
    # the real display, and a bare show() would flash this window on screen.
    win.setAttribute(Qt.WA_DontShowOnScreen)
    win.setCentralWidget(QWidget())
    win.resize(width, height)

    def dock(name, area):
        d = QDockWidget(name)
        d.setObjectName(name)
        d.setWidget(QWidget())
        win.addDockWidget(area, d)
        return d

    area = Qt.RightDockWidgetArea if side == "right" else Qt.LeftDockWidgetArea
    img = dock(image_dock, area)

    win.show()
    app.processEvents()
    win.resizeDocks([img], [int(width * split)], Qt.Horizontal)
    app.processEvents()
    return bytes(win.saveState()).hex()


def primary_screen_size():
    import os
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return None
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    g = app.primaryScreen().availableGeometry()
    return g.width(), g.height()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("template")
    p.add_argument("out")
    p.add_argument("--image-dock", default="Camera",
                   help="Name of the Image display whose panel takes the left half.")
    p.add_argument("--split", type=float, default=0.5,
                   help="Fraction of the window width given to the camera.")
    p.add_argument("--camera-side", choices=("right", "left"), default="right")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    a = p.parse_args()

    if a.width and a.height:
        w, h = a.width, a.height
    else:
        size = primary_screen_size()
        if size is None:
            sys.exit("no display found — pass --width and --height")
        w, h = size

    with open(a.template) as fh:
        cfg = yaml.safe_load(fh)

    names = [d.get("Name") for d in cfg["Visualization Manager"]["Displays"]]
    if a.image_dock not in names:
        sys.exit(f"template has no display named {a.image_dock!r} (has {names})")

    state = build_state(w, h, a.image_dock, a.split, a.camera_side)
    geo = cfg.setdefault("Window Geometry", {})
    geo.update({
        "Width": w, "Height": h, "X": 0, "Y": 0,
        "Hide Left Dock": False, "Hide Right Dock": False,
        "QMainWindow State": state,
    })
    geo.setdefault(a.image_dock, {"collapsed": False})

    with open(a.out, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, width=1000)
    print(f"{a.out}: {w}x{h}, camera '{a.image_dock}' {a.split:.0%} on the {a.camera_side}")


if __name__ == "__main__":
    main()
