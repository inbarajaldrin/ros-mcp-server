#!/usr/bin/env python3
"""Render the milestone ladder (compliant_insertion_studio/docs/LADDER.json) as a phone-readable PNG.

    python3 ablations/ladder_image.py [OUT.png] [--width 1080]

One card per rung: a status pill (done / active / next / blocked), the id and title, the note wrapped. Grouped by
track. Made for sending to the operator's phone (text-me / Dimelab), 2026-10-21.
"""
import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / 'compliant_insertion_studio' / 'docs' / 'LADDER.json'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
STATUS = {  # fill, text
    'done':    ((34, 139, 84), (255, 255, 255)),
    'active':  ((37, 99, 235), (255, 255, 255)),
    'next':    ((209, 213, 219), (55, 65, 81)),
    'blocked': ((220, 38, 38), (255, 255, 255)),
}
BG, CARD, INK, MUTED, RULE = (246, 247, 249), (255, 255, 255), (17, 24, 39), (75, 85, 99), (229, 231, 235)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default=str(REPO / 'ablations' / 'logs' / 'ladder.png'))
    ap.add_argument('--width', type=int, default=1080)
    a = ap.parse_args()
    d = json.load(open(LADDER))
    W, M, P = a.width, 36, 24                      # width, outer margin, card padding
    f_title, f_track = ImageFont.truetype(FONT_B, 46), ImageFont.truetype(FONT_B, 34)
    f_head, f_pill = ImageFont.truetype(FONT_B, 30), ImageFont.truetype(FONT_B, 22)
    f_note, f_small = ImageFont.truetype(FONT, 25), ImageFont.truetype(FONT, 22)
    chars = int((W - 2 * M - 2 * P) / (f_note.getlength('abcdefghijklmnopqrstuvwxyz') / 26))

    # lay out first (measure), then draw
    blocks = []
    for t in d['tracks']:
        rungs = t.get('rungs', t.get('items', []))
        blocks.append(('track', t['name'], rungs))
        for r in rungs:
            note = textwrap.wrap(r.get('note', ''), chars)[:9]
            title = textwrap.wrap(f"{r['id']}  {r['title']}", int(chars * 0.72))
            blocks.append(('rung', r, title, note))
    h = M + 70 + 46
    for b in blocks:
        h += 70 if b[0] == 'track' else (P * 2 + 40 * len(b[2]) + 33 * len(b[3]) + 14)
    h += M
    img = Image.new('RGB', (W, h), BG); g = ImageDraw.Draw(img)
    y = M
    g.text((M, y), 'Milestone ladder', font=f_title, fill=INK); y += 62
    g.text((M, y), f"updated {d.get('updated', '')}   ·   goal: {d.get('goal', '')[:60]}", font=f_small, fill=MUTED); y += 46
    for b in blocks:
        if b[0] == 'track':
            _, name, rungs = b
            n_done = sum(r['status'] == 'done' for r in rungs)
            g.text((M, y + 16), name, font=f_track, fill=INK)
            g.text((W - M - f_small.getlength(f"{n_done}/{len(rungs)} done"), y + 26),
                   f"{n_done}/{len(rungs)} done", font=f_small, fill=MUTED)
            y += 70
            continue
        _, r, title, note = b
        ch = P * 2 + 40 * len(title) + 33 * len(note)
        fill, tc = STATUS.get(r['status'], STATUS['next'])
        g.rounded_rectangle((M, y, W - M, y + ch), radius=16, fill=CARD, outline=RULE, width=2)
        g.rounded_rectangle((M, y, M + 10, y + ch), radius=5, fill=fill)
        lab = r['status'].upper(); pw = f_pill.getlength(lab) + 28
        g.rounded_rectangle((W - M - P - pw, y + P - 2, W - M - P, y + P + 32), radius=16, fill=fill)
        g.text((W - M - P - pw + 14, y + P + 1), lab, font=f_pill, fill=tc)
        yy = y + P
        for line in title:
            g.text((M + P + 8, yy), line, font=f_head, fill=INK); yy += 40
        for line in note:
            g.text((M + P + 8, yy), line, font=f_note, fill=MUTED); yy += 33
        y += ch + 14
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out)
    print(a.out, img.size)


if __name__ == '__main__':
    main()
