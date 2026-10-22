"""Save one still wrist-camera frame (emitter off, brightened) + TCP pose: python3 look.py OUT.png [gain]"""
import sys, subprocess, time, numpy as np, cv2, json
REPO='/home/aldrin/Documents/ros-mcp-server-verified'; sys.path[:0]=[REPO, REPO+'/'+'abl'+'ations']
from compliant_insertion_studio.wrapper.marker_yaw import still_frame, _tcp_pose_now
out=sys.argv[1]; gain=float(sys.argv[2]) if len(sys.argv)>2 else 3.0
def em(on): subprocess.run(['ros2','param','set','/camera/camera','depth_module.emitter_enabled','1' if on else '0'],capture_output=True,timeout=10); time.sleep(0.8)
em(False)
img,_=still_frame(n_avg=6); p,q=_tcp_pose_now()
em(True)
if img is None: print('no still frame'); sys.exit(1)
v=np.clip(img.astype(np.float32)*gain,0,255).astype(np.uint8)
cv2.imwrite(out,v); print(json.dumps({'tcp_mm':[round(1000*x,1) for x in p],'quat':[round(float(x),4) for x in q],'out':out}))
