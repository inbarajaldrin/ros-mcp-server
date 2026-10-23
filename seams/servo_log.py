"""Servo sessions parsed from the replay logs (the hover servo's own lines) - test data for seams.servo_law."""
import re

PART = re.compile(r"\[servo\] (\w+) round (\d+): part \[([-\d.]+), ([-\d.]+)\] mm from over its socket, yaw ([-\d.]+|None) deg"
                  r".*?spread \[([-\d.]+), ([-\d.]+)\] px")
FAR = re.compile(r"\[servo\] (\w+) round (\d+): left-edge spread .* far-edge component only \[([-\d.]+), ([-\d.]+)\] mm")
TURN = re.compile(r"\[servo\] (\w+): wrist turned ([-+\d.]+) deg \(image yaw\)")
AIM = re.compile(r"\[servo\] (\w+): aim moved by \(([-+\d.]+), ([-+\d.]+)\) mm")
NOTAPP = re.compile(r"\[servo\] (\w+): correction ([\d.]+) mm is more than the reads can be off")
CLEAR = re.compile(r"\[servo\] (\w+) round (\d+): .* measuring from the clear height instead")
LOW = re.compile(r"\[servo\] (\w+): touched down .* measuring from 2 mm above the contact")
STOPS = re.compile(r"\[servo\] (\w+) round (\d+): (base not read in the frame|not measured|fit spread)")
YAWSTOP = re.compile(r"\[servo\] (\w+): image yaw .* stopping the servo")
OUTLINED = re.compile(r"\[servo\] (\w+): held part outlined")


def sessions(text, source=''):
    """[{obj, events: [...], aim: (x, y) mm | None, not_applied: bool}] - one per hover_servo call that measured."""
    out, cur = [], None
    for line in text.splitlines():
        m = OUTLINED.search(line)
        if m:
            cur = dict(obj=m.group(1), events=[], aim=None, not_applied=False, source=source); continue
        if cur is None:
            continue
        for rx, kind in ((CLEAR, 'clear'), (LOW, 'low'), (FAR, 'far'), (PART, 'part'), (TURN, 'turn'),
                         (STOPS, 'stop'), (YAWSTOP, 'yawstop')):
            m = rx.search(line)
            if m and m.group(1) == cur['obj']:
                g = m.groups()
                if kind == 'part':
                    cur['events'].append(dict(kind='part', k=int(g[1]), d_mm=[float(g[2]), float(g[3])],
                                              yaw=None if g[4] == 'None' else float(g[4]), spread=[float(g[5]), float(g[6])]))
                elif kind == 'far':
                    cur['events'].append(dict(kind='far', k=int(g[1]), far_mm=[float(g[2]), float(g[3])]))
                elif kind == 'turn':
                    cur['events'].append(dict(kind='turn', deg=float(g[1])))
                elif kind == 'stop':
                    cur['events'].append(dict(kind='stop', k=int(g[1]), what=g[2]))
                else:
                    cur['events'].append(dict(kind=kind))
                break
        m = AIM.search(line)
        if m and m.group(1) == cur['obj']:
            cur['aim'] = [float(m.group(2)), float(m.group(3))]; out.append(cur); cur = None; continue
        m = NOTAPP.search(line)
        if m and m.group(1) == cur['obj']:
            cur['not_applied'] = True; out.append(cur); cur = None
    return out
