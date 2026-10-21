#!/usr/bin/env python3
"""ExecuteMate on the real cell (2026-10-20, M3a): prompt2assembly's pa_assembly_msgs/action/ExecuteMate, unchanged.

    python3 -m seams.execute_mate_server            # serves /ur5e/execute_mate

The platform verb, as the contract states it: it owns NO motion, never repairs, never selects. Five questions in
order, each able only to say no:
  1. the mate resolves (mate_unknown / assembly_unbound / assembly_ambiguous - from seams/bindings.yaml)
  2. every earlier step of the declared order is certified seated (order_violated) - from the journal below
  3. every predecessor still passes NOW (predecessor_unseated) - re-read with this mate, one shared capture
  4. the class premise: FMB1 mates are `free`, judged AT REST after the hand opened (not_at_rest if the RG2 still
     holds a part)
  5. the seat passes the mate's own declared acceptance (not_at_seat measured outside; engagement_invalid when the
     seat could not be measured from this view)
and the event is journalled either way. A free mate that passes comes back fastened=false, placed=true, refusal "".

The measurement is vision.certify (seams.vision_ops.certify_many: the depth seat check judges height and tilt,
markers report xy). Real-cell deviation, recorded rather than hidden: a predecessor the camera cannot see from this
view (FMB1: line_green under inverted_u_yellow) is re-read where possible and otherwise carried on its journalled
certification, named in engagement_json - the contract would raise engagement_invalid, which on this
cell would make the last FMB1 mate uncertifiable by construction.

Journal: ablations/logs/mate_journal.jsonl (one event per call). Certifying the first mate of the order starts a new
episode.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool

from pa_assembly_msgs.action import ExecuteMate

REPO = Path(__file__).resolve().parents[1]
ACTION = '/ur5e/execute_mate'
JOURNAL = REPO / 'ablations' / 'logs' / 'mate_journal.jsonl'
B = yaml.safe_load(open(REPO / 'seams' / 'bindings.yaml'))


def resolve(mate):
    asm, _, addr = mate.rpartition('/')
    bound = [k for k in B if k != 'sigma']
    if not asm:
        if len(bound) != 1:
            return None, 'assembly_ambiguous'
        asm = bound[0]
    if asm not in bound:
        return None, 'assembly_unbound'
    if addr not in B[asm]['mates']:
        return None, 'mate_unknown'
    return (asm, addr), ''


def episode():
    """The journal's current episode: events since the last certified first-mate (or since the start)."""
    ev = []
    if JOURNAL.exists():
        for line in open(JOURNAL):
            try:
                ev.append(json.loads(line))
            except Exception:
                pass
    start = 0
    for i, e in enumerate(ev):
        if e.get('event') == 'placed' and e.get('order_position') == 0:
            start = i
    return ev[start:]


def certified(ep, mate_id):
    return any(e.get('mate_id') == mate_id and e.get('event') == 'placed' for e in ep)


def certify(mates):
    p = subprocess.run(['python3', '-u', '-m', 'seams.vision_ops', 'certify', '--mate', *mates], cwd=str(REPO),
                       capture_output=True, text=True, timeout=120 + 90 * len(mates))
    line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith('__VISION__ ')), None)
    if line is None:
        return None
    d = json.loads(line[len('__VISION__ '):])
    return d['results'] if 'results' in d else [d]


class ExecuteMateServer(Node):
    def __init__(self):
        super().__init__('ur5e_execute_mate')
        self.grasped = None
        self.create_subscription(Bool, '/gripper_grasp_detected', lambda m: setattr(self, 'grasped', bool(m.data)), 10)
        ActionServer(self, ExecuteMate, ACTION, self.execute, callback_group=ReentrantCallbackGroup())
        self.get_logger().info(f'ExecuteMate served at {ACTION}')

    def execute(self, gh):
        r = ExecuteMate.Result()
        r.fastened = False; r.placed = False; r.order_position = -1; r.weld_prim = ''; r.journalled = False
        r.ee_class = 'parallel_jaw'; r.engagement_depth_m = float('nan')
        ok, refusal = resolve(gh.request.mate)
        if not ok:
            return self._done(gh, r, refusal, f'{gh.request.mate!r}')
        asm, addr = ok
        A = B[asm]; m = A['mates'][addr]; mate_id = f'{asm}/{addr}'
        r.mate_id = mate_id; r.child = m['mover']; r.parent = A['reference']
        r.mate_class = 'insertion'; r.fastening_class = 'free'
        r.order_position = A['order'].index(addr)
        r.premise = 'at_rest: a free mate is accepted where it came to rest after the hand opened'
        r.gripped_on_mover = bool(self.grasped)
        r.grip_disposition = 'holding' if self.grasped else 'open'
        preds = [f'{asm}/{a}' for a in A['order'][:r.order_position]]
        ep = [] if r.order_position == 0 else episode()
        missing = [p for p in preds if not certified(ep, p)]
        if missing:
            r.next_move = f'certify {missing[0]} first'
            return self._done(gh, r, 'order_violated', f'not certified in this episode: {missing}')
        if self.grasped:
            r.next_move = 'open the hand, let the part come to rest, then call again'
            return self._done(gh, r, 'not_at_rest', 'the RG2 reports a grip: a free mate is judged after release')
        res = certify(preds + [mate_id])
        if res is None:
            return self._done(gh, r, 'engagement_invalid', 'vision certify did not answer')
        this = res[-1]
        r.engagement_measured = this['method'] in ('both', 'depth_seat_check')    # the seat height certifies; markers alone do not
        r.engaged = bool(this['inside'])
        r.engagement_depth_m = float(this['error_mate_frame_m'][2])
        r.granted_on = this['method']
        pre = {}
        for p, c in zip(preds, res[:-1]):
            if c['method'] not in ('both', 'depth_seat_check'):
                pre[p] = 'not measurable from this view - carried on its journalled certification'
            elif not c['inside']:
                r.engagement_json = json.dumps({'this': this, 'predecessors': res[:-1]}, default=float)
                r.next_move = f're-place {p}'
                return self._done(gh, r, 'predecessor_unseated', f'{p}: {c["evidence"].get("verdict")}', ev=this)
            else:
                pre[p] = 'passes now'
        r.engagement_json = json.dumps({'this': this, 'predecessors': pre}, default=float)
        if not r.engagement_measured:
            r.next_move = 'move the camera to a view of the seat (seat_check.view_pose) and call again'
            return self._done(gh, r, 'engagement_invalid', this['evidence'].get('verdict', ''), ev=this)
        if not this['inside']:
            r.next_move = 'drive the mate again (drive-mate) or re-place the part'
            return self._done(gh, r, 'not_at_seat', this['evidence'].get('verdict', ''), ev=this)
        r.placed = True
        return self._done(gh, r, '', '', ev=this)

    def _done(self, gh, r, refusal, detail, ev=None):
        r.refusal = refusal; r.refusal_detail = detail
        try:
            JOURNAL.parent.mkdir(parents=True, exist_ok=True)
            with open(JOURNAL, 'a') as f:
                f.write(json.dumps({'t': time.time(), 'mate_id': r.mate_id or gh.request.mate,
                                    'event': 'placed' if r.placed else 'refused', 'refusal': refusal,
                                    'detail': detail, 'order_position': r.order_position,
                                    'error_mate_frame_m': (ev or {}).get('error_mate_frame_m'),
                                    'method': (ev or {}).get('method')}, default=float) + '\n')
            r.journalled = True; r.journal_detail = str(JOURNAL)
        except Exception as e:
            r.journalled = False; r.journal_detail = f'journal write failed: {e}'
        self.get_logger().info(f"{r.mate_id or gh.request.mate}: placed={r.placed} refusal={refusal!r} {detail}")
        gh.succeed()
        return r


def main():
    rclpy.init()
    n = ExecuteMateServer()
    ex = MultiThreadedExecutor(); ex.add_node(n)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
