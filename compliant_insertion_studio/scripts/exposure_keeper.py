#!/usr/bin/env python3
"""Keep the wrist D415's colour image usable as the lab lights change.

The camera launch keeps `rgb_camera.auto_exposure_priority:=false` for 15 fps. With the lights off, that caps the auto exposure
and the colour image goes black (brightness 5-8/255; markers lost). What worked by hand (10-21, 11-20) is manual exposure 8000 + gain
128. With the lights back on, that manual setting leaves every frame white (10-25). This node does both switches by itself:

  AUTO   -> MANUAL  when the frame mean stays below DARK for HOLD samples: gain 128, auto exposure off, exposure 8000.
  MANUAL            each sample outside [LOW, HIGH]: scale the exposure toward TARGET, clamped to [EXP_MIN, EXP_MAX].
  MANUAL -> AUTO    when the exposure is at EXP_MIN and the frame still stays above HIGH for HOLD samples: auto on, gain 64.

It samples every PERIOD_S, so it never fights the camera's own auto exposure. Depth is unaffected: the D415 uses its IR projector.
Transitions are logged to /tmp/exposure_keeper.log. Started by stack_up.sh; FMB_EXPOSURE_KEEPER=0 skips it.

    python3 compliant_insertion_studio/scripts/exposure_keeper.py [--once]
"""
import argparse
import time

DARK, LOW, HIGH, TARGET = 20.0, 50.0, 170.0, 90.0
EXP_MIN, EXP_MAX = 156, 8000
GAIN_DARK, GAIN_LIT = 128, 64
HOLD = 3
PERIOD_S = 2.0
NODE = '/camera/camera'
IMG = '/camera/camera/color/image_raw'


def decide(state, mean, exposure, streak):
    """One sample. Returns (new_state, params_to_set: dict, new_streak). Pure, for the tests."""
    if state == 'auto':
        streak = streak + 1 if mean < DARK else 0
        if streak >= HOLD:
            return 'manual', {'rgb_camera.gain': GAIN_DARK, 'rgb_camera.enable_auto_exposure': False,
                              'rgb_camera.exposure': EXP_MAX}, 0
        return 'auto', {}, streak
    # manual
    if exposure <= EXP_MIN and mean > HIGH:
        streak += 1
        if streak >= HOLD:
            return 'auto', {'rgb_camera.enable_auto_exposure': True, 'rgb_camera.gain': GAIN_LIT}, 0
        return 'manual', {}, streak
    if mean < LOW or mean > HIGH:
        new = int(min(EXP_MAX, max(EXP_MIN, exposure * TARGET / max(mean, 1.0))))
        if new != exposure:
            return 'manual', {'rgb_camera.exposure': new}, 0
    return 'manual', {}, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true', help='print one sample and the decision, change nothing')
    a = ap.parse_args()
    import numpy as np
    import rclpy
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import GetParameters, SetParameters
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node('exposure_keeper')
    last = {}
    node.create_subscription(Image, IMG, lambda m: last.__setitem__('m', m), 1)
    get = node.create_client(GetParameters, NODE + '/get_parameters')
    put = node.create_client(SetParameters, NODE + '/set_parameters')

    def call(client, req):
        if not client.wait_for_service(timeout_sec=5.0):
            return None
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        return fut.result()

    def read_state():
        r = call(get, GetParameters.Request(names=['rgb_camera.enable_auto_exposure', 'rgb_camera.exposure']))
        if r is None or len(r.values) < 2:
            return None, None
        return ('auto' if r.values[0].bool_value else 'manual'), int(r.values[1].integer_value)

    def write(params):
        ps = []
        for k, v in params.items():
            pv = (ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=v) if isinstance(v, bool)
                  else ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(v)))
            ps.append(Parameter(name=k, value=pv))
        r = call(put, SetParameters.Request(parameters=ps))
        return r is not None and all(x.successful for x in r.results)

    def log(text):
        line = f"{time.strftime('%F %T')} {text}"
        print(line, flush=True)
        with open('/tmp/exposure_keeper.log', 'a') as fh:
            fh.write(line + '\n')

    streak = 0
    while rclpy.ok():
        t0 = time.time()
        last.pop('m', None)
        while 'm' not in last and time.time() - t0 < 5.0:
            rclpy.spin_once(node, timeout_sec=0.1)
        state, exposure = read_state()
        if 'm' not in last or state is None:
            if a.once:
                print('no frame or no camera parameters'); return 1
            time.sleep(PERIOD_S); continue
        m = last['m']
        mean = float(np.frombuffer(m.data, np.uint8)[::97].mean())   # subsampled: cheap, plenty for a mean
        new_state, params, streak = decide(state, mean, exposure, streak)
        if a.once:
            print(f'brightness {mean:.1f}, state {state}, exposure {exposure}, would set {params or "nothing"}')
            return 0
        if params:
            ok = write(params)
            log(f'brightness {mean:.1f}: {state} -> {new_state}, set {params} ({"ok" if ok else "FAILED"})')
        time.sleep(max(0.0, PERIOD_S - (time.time() - t0)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
