"""Small, ROS-independent force-filtering helpers."""

from statistics import median


def median_wrench_window(samples, window_s, *, min_span_fraction=0.6):
    """Return a six-axis median over a trailing timestamp window.

    Each sample is ``(stamp_s, fx, fy, fz, tx, ty, tz)``. Refuse a
    partial window so a short impulse cannot become the whole population.
    """
    rows = list(samples)
    if len(rows) < 3 or window_s <= 0:
        return None
    latest = float(rows[-1][0])
    selected = [row for row in rows if float(row[0]) >= latest - window_s]
    if len(selected) < 3:
        return None
    span_s = float(selected[-1][0]) - float(selected[0][0])
    if span_s < window_s * min_span_fraction:
        return None
    return tuple(float(median(row[axis] for row in selected)) for axis in range(1, 7))
