"""
Compare two compliant-insert runs side by side with overlay plots.

Usage:
    python3 compare_runs.py <baseline_csv> <test_csv> [--out <html>]

Default output: /tmp/compare_runs.html  (drop into a browser)

Plots emitted (each as a subplot in the same HTML):
  1. xy trajectory (with hole_xy_prior marker, contact-point marker)
  2. tcp_z(t)            — descent profile
  3. distance-to-hole(t) — convergence vs drift
  4. fz(t)               — contact force timeline
  5. |F_lat|(t)          — lateral wrench magnitude
  6. |T_lat|(t)          — torque magnitude

The two runs are color-coded (baseline = green, test = red) and overlaid
where it makes sense; the trajectory is plotted with arrows so direction
is visible.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_run(csv_path: Path) -> dict:
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    't': float(r['t_s']),
                    'phase': r['phase'],
                    'x': float(r['tcp_x']), 'y': float(r['tcp_y']), 'z': float(r['tcp_z']),
                    'fx': float(r['fx']), 'fy': float(r['fy']), 'fz': float(r['fz']),
                    'tx': float(r['tx']), 'ty': float(r['ty']),
                })
            except (KeyError, ValueError):
                pass
    meta_path = csv_path.with_suffix('.meta.json' if csv_path.suffix == '.csv' else '')
    meta_path = Path(str(csv_path).replace('.csv', '.meta.json'))
    meta = {}
    if meta_path.exists():
        try:
            meta = json.load(meta_path.open())
        except Exception:
            pass
    return {'name': csv_path.name, 'rows': rows, 'meta': meta}


def slice_active(run: dict) -> dict:
    """Returns ACTIVE-only data + contact info."""
    active = [r for r in run['rows'] if r['phase'] == 'ACTIVE']
    if not active:
        return {**run, 'active': [], 'contact_idx': None}
    contact_idx = next((i for i, r in enumerate(active) if r['fz'] > 6.0), None)
    return {**run, 'active': active, 'contact_idx': contact_idx}


def add_run_traces(fig, run: dict, color: str, label: str, hole_xy: tuple | None):
    a = run['active']
    if not a:
        return
    ts = [r['t'] - a[0]['t'] for r in a]   # zero at ACTIVE start
    xs = [r['x'] for r in a]
    ys = [r['y'] for r in a]
    zs = [r['z'] for r in a]
    fzs = [r['fz'] for r in a]
    F_lats = [math.hypot(r['fx'], r['fy']) for r in a]
    T_lats = [math.hypot(r['tx'], r['ty']) for r in a]

    contact_t = ts[run['contact_idx']] if run['contact_idx'] is not None else None
    contact_xy = (xs[run['contact_idx']], ys[run['contact_idx']]) if run['contact_idx'] is not None else None

    # 1. xy trajectory
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='lines', name=f'{label} xy traj',
        line=dict(color=color, width=2),
        legendgroup=label, showlegend=True,
    ), row=1, col=1)
    if contact_xy is not None:
        fig.add_trace(go.Scatter(
            x=[contact_xy[0]], y=[contact_xy[1]], mode='markers',
            name=f'{label} contact', marker=dict(color=color, size=14, symbol='star'),
            legendgroup=label, showlegend=True,
        ), row=1, col=1)
    # final position
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]], mode='markers',
        name=f'{label} final', marker=dict(color=color, size=12, symbol='x'),
        legendgroup=label, showlegend=True,
    ), row=1, col=1)

    # 2. tcp_z(t)
    fig.add_trace(go.Scatter(
        x=ts, y=zs, mode='lines', name=f'{label} z',
        line=dict(color=color, width=2),
        legendgroup=label, showlegend=False,
    ), row=1, col=2)

    # 3. distance to hole
    if hole_xy is not None:
        dists = [math.hypot(x - hole_xy[0], y - hole_xy[1]) * 1000 for x, y in zip(xs, ys)]
        fig.add_trace(go.Scatter(
            x=ts, y=dists, mode='lines', name=f'{label} dist→hole',
            line=dict(color=color, width=2),
            legendgroup=label, showlegend=False,
        ), row=2, col=1)

    # 4. fz(t)
    fig.add_trace(go.Scatter(
        x=ts, y=fzs, mode='lines', name=f'{label} fz',
        line=dict(color=color, width=1.5),
        legendgroup=label, showlegend=False,
    ), row=2, col=2)

    # 5. F_lat magnitude
    fig.add_trace(go.Scatter(
        x=ts, y=F_lats, mode='lines', name=f'{label} |F_lat|',
        line=dict(color=color, width=1.5),
        legendgroup=label, showlegend=False,
    ), row=3, col=1)

    # 6. T_lat magnitude
    fig.add_trace(go.Scatter(
        x=ts, y=T_lats, mode='lines', name=f'{label} |T_lat|',
        line=dict(color=color, width=1.5),
        legendgroup=label, showlegend=False,
    ), row=3, col=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline', help='Baseline CSV (e.g., operator-demo success)')
    ap.add_argument('test', help='Test CSV (e.g., automated attempt)')
    ap.add_argument('--hole-xy', nargs=2, type=float, default=None,
                    metavar=('X', 'Y'), help='Override hole xy in meters; '
                    'else uses test run meta hole_xy_prior or contact xy')
    ap.add_argument('--out', default='/tmp/compare_runs.html')
    ap.add_argument('--baseline-label', default='operator demo')
    ap.add_argument('--test-label', default='today auto')
    args = ap.parse_args()

    baseline = slice_active(load_run(Path(args.baseline)))
    test = slice_active(load_run(Path(args.test)))

    # Hole xy: prefer explicit arg, then test run's hole_xy_prior or contact xy
    hole_xy = None
    if args.hole_xy is not None:
        hole_xy = tuple(args.hole_xy)
    else:
        hxp = test['meta'].get('hole_xy_prior')
        if hxp:
            hole_xy = (float(hxp[0]), float(hxp[1]))
        elif baseline['active'] and baseline['contact_idx'] is not None:
            # Use baseline's FINAL xy as the truth-ground hole position
            last = baseline['active'][-1]
            hole_xy = (last['x'], last['y'])

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'xy trajectory (top-down)', 'tcp_z over time',
            'distance to hole (mm)', 'fz over time (N)',
            '|F_lat| over time (N)', '|T_lat| over time (Nm)',
        ),
        specs=[
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}],
        ],
    )

    add_run_traces(fig, baseline, color='green', label=args.baseline_label, hole_xy=hole_xy)
    add_run_traces(fig, test, color='red', label=args.test_label, hole_xy=hole_xy)

    if hole_xy is not None:
        fig.add_trace(go.Scatter(
            x=[hole_xy[0]], y=[hole_xy[1]], mode='markers',
            name='hole', marker=dict(color='gold', size=18, symbol='circle-open',
                                     line=dict(color='black', width=2)),
        ), row=1, col=1)

    # Equal-aspect xy
    fig.update_yaxes(scaleanchor='x', scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text='x (m)', row=1, col=1)
    fig.update_yaxes(title_text='y (m)', row=1, col=1)
    fig.update_xaxes(title_text='t since ACTIVE (s)', row=1, col=2)
    fig.update_yaxes(title_text='z (m)', row=1, col=2)
    fig.update_xaxes(title_text='t since ACTIVE (s)', row=2, col=1)
    fig.update_yaxes(title_text='dist (mm)', row=2, col=1)
    fig.update_xaxes(title_text='t since ACTIVE (s)', row=2, col=2)
    fig.update_yaxes(title_text='fz (N)', row=2, col=2)
    fig.update_xaxes(title_text='t since ACTIVE (s)', row=3, col=1)
    fig.update_yaxes(title_text='|F_lat| (N)', row=3, col=1)
    fig.update_xaxes(title_text='t since ACTIVE (s)', row=3, col=2)
    fig.update_yaxes(title_text='|T_lat| (Nm)', row=3, col=2)

    title = f'Compare: <b style="color:green">{args.baseline_label}</b> vs <b style="color:red">{args.test_label}</b><br>'
    title += f'<sup>baseline: {Path(args.baseline).name}<br>test: {Path(args.test).name}'
    if hole_xy is not None:
        title += f'<br>hole xy = ({hole_xy[0]:.4f}, {hole_xy[1]:.4f}) m</sup>'

    fig.update_layout(
        title=title,
        height=1200, width=1500,
        hovermode='closest',
    )

    fig.write_html(args.out)
    print(f'wrote {args.out}')

    # Print quantitative summary
    print('\n=== summary ===')
    for label, run in (('baseline', baseline), ('test', test)):
        a = run['active']
        if not a or run['contact_idx'] is None:
            print(f'{label}: no ACTIVE samples or no contact'); continue
        ci = run['contact_idx']
        descent_mm = (a[ci]['z'] - a[-1]['z']) * 1000
        lat_travel_mm = math.hypot(a[-1]['x'] - a[ci]['x'], a[-1]['y'] - a[ci]['y']) * 1000
        F_lats = [math.hypot(r['fx'], r['fy']) for r in a[ci:]]
        T_lats = [math.hypot(r['tx'], r['ty']) for r in a[ci:]]
        d_hole_initial = math.hypot(a[ci]['x']-hole_xy[0], a[ci]['y']-hole_xy[1])*1000 if hole_xy else None
        d_hole_final = math.hypot(a[-1]['x']-hole_xy[0], a[-1]['y']-hole_xy[1])*1000 if hole_xy else None
        print(f'{label:>10}  contact_z={a[ci]["z"]:.4f} final_z={a[-1]["z"]:.4f} '
              f'descent={descent_mm:.1f}mm  lat_travel={lat_travel_mm:.1f}mm  '
              f'|F_lat|max={max(F_lats):.1f}N  |T_lat|max={max(T_lats):.3f}Nm')
        if hole_xy:
            print(f'           dist_to_hole: contact={d_hole_initial:.1f}mm → final={d_hole_final:.1f}mm '
                  f'(Δ={d_hole_final-d_hole_initial:+.1f}mm)')


if __name__ == '__main__':
    main()
