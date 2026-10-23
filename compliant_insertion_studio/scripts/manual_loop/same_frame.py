"""One-frame pocket-vs-part measurement at the current pose (2 mm over the rim works best):
  base from its markers in THIS frame (tilt from the chain, yaw searched, translation solved) -> pocket projection;
  part = its own far/left edges in THIS frame; socket_shift_world -> d_world (real - target) and image yaw.
python3 same_frame.py OUT.png BASE_YAW_GUESS_DEG [TURN_DEG]   (TURN_DEG: turn the wrist about world z first)"""
import sys, subprocess, time, json, numpy as np, cv2
REPO='/home/aldrin/Documents/ros-mcp-server-verified'; sys.path[:0]=[REPO, REPO+'/'+'abl'+'ations']
from scipy.spatial.transform import Rotation as R
from seat_check import X_CAM
from handeye_icp_survey import T as _T
from compliant_insertion_studio.wrapper.marker_yaw import still_frame, _tcp_pose_now, _intrinsics, _layout, _detect, socket_shift_world
from compliant_insertion_studio.wrapper.align_at_hover import turn_wrist
out=sys.argv[1]; byaw=float(sys.argv[2]); turn=float(sys.argv[3]) if len(sys.argv)>3 else 0.0
if abs(turn)>0.01: print('turn_wrist', turn, turn_wrist(turn)); time.sleep(0.5)
if len(sys.argv)>5 and (abs(float(sys.argv[4]))>0.01 or abs(float(sys.argv[5]))>0.01):
    dx,dy=float(sys.argv[4])/1000,float(sys.argv[5])/1000; p0,q0=_tcp_pose_now()
    r=subprocess.run([sys.executable,'-u','-m','primitives.move_cartesian','--mode','real','--xyz',f"{p0[0]+dx:.5f}",f"{p0[1]+dy:.5f}",f"{p0[2]:.5f}",
        '--quat',*[f"{v:.6f}" for v in q0],'--min-z','0.15','--max-force','4','--min-duration','3'],cwd=REPO,capture_output=True,text=True,timeout=90)
    print('move xy mm', 1000*dx, 1000*dy, r.stdout.splitlines()[-2][:90] if r.stdout else r.stderr[-200:]); time.sleep(0.5)
def em(on): subprocess.run(['ros2','param','set','/camera/camera','depth_module.emitter_enabled','1' if on else '0'],capture_output=True,timeout=10); time.sleep(0.8)
em(False); img,_=still_frame(n_avg=6); p,q=_tcp_pose_now(); em(True)
K,D=_intrinsics(); Twc=_T(p,q)@X_CAM; Tcw=np.linalg.inv(Twc)
dn,lay=_layout('base1'); corners,ids=_detect(img,dn); det={int(i):c.reshape(-1,2) for c,i in zip(corners,ids)} if ids is not None else {}
bid=[i for i in det if i in lay]
def c3d(i):
    t,Rm,edge=lay[i]; h=edge/2; return np.array([Rm@np.array(lc)+t for lc in ([-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0])])
pts=np.concatenate([c3d(i) for i in bid]); pix=np.concatenate([det[i] for i in bid])
nrm=cv2.undistortPoints(np.array(pix,np.float32).reshape(-1,1,2),K,D).reshape(-1,2)
def solve(Rc):
    A,b=[],[]
    for X,(u,v) in zip(pts,nrm):
        Xc=Rc@X; A+= [[1,0,-u],[0,1,-v]]; b+=[u*Xc[2]-Xc[0], v*Xc[2]-Xc[1]]
    t=np.linalg.lstsq(np.array(A),np.array(b),rcond=None)[0]
    uv,_=cv2.projectPoints(((Rc@pts.T).T+t).astype(np.float32),np.zeros(3),np.zeros(3),K,D)
    return t, float(np.linalg.norm(uv.reshape(-1,2)-pix,axis=1).mean())
best=None
for dz in np.arange(-6,6.01,0.05):
    Rc=Tcw[:3,:3]@R.from_euler('z',byaw+dz,degrees=True).as_matrix(); t,rp=solve(Rc)
    if best is None or rp<best[1]: best=(t,rp,dz)
tb,rb,dzb=best; yaw_m=byaw+dzb; bw=Twc[:3,:3]@tb+Twc[:3,3]; bq=R.from_euler('z',yaw_m,degrees=True).as_quat()
# part edges: far = first row (from y=400 down) where R<60 for 3 rows, per column; left = per row, scanning left from
# x=700 until R<20 (finger) or R>75 (base) - the part top reads R 25..60 raw
g=img.astype(int); Rch=g[:,:,2]
far=[]
for x in range(660,881,8):
    col=Rch[400:600,x]; idx=[i for i in range(197) if col[i]<60 and col[i+1]<60 and col[i+2]<60]
    if idx: far.append((x,400+idx[0]))
left=[]
for y in range(530,601,5):
    x=700
    while x>560 and 20<=Rch[y,x]<=75: x-=1
    if 580<x<699: left.append((x+1,y))
if len(far)>=5:
    my=np.median([y for _,y in far]); far=[(x,y) for x,y in far if abs(y-my)<=6]
if len(left)>=5:
    mx=np.median([x for x,_ in left]); left=[(x,y) for x,y in left if abs(x-mx)<=6]
outline={'far':far,'left':left}
import os
if os.environ.get('OUTLINE_JSON'):
    oj=json.load(open(os.environ['OUTLINE_JSON'])); outline={'far':[tuple(v) for v in oj['far']],'left':[tuple(v) for v in oj['left']]}; far=outline['far']
lift=p[2]-0.1867
d,info=socket_shift_world('u_brown','base1',(bw,bq),lift,outline,img=img,tcp_pose=(p,q),debug_path=out)
# direct image slopes: base marker line (0->3 centres) vs the part's far edge
slope_part=np.polyfit([x for x,_ in far],[y for _,y in far],1)[0] if len(far)>=5 else None
mline=None
if 0 in det and 3 in det:
    c0,c3=det[0].mean(0),det[3].mean(0); mline=float(np.degrees(np.arctan2(c3[1]-c0[1],c3[0]-c0[0])))
base_slope=None
if mline is not None:
    a=(mline+180)%360-180; base_slope=-( (abs(a)-180) if abs(a)>90 else a ) if a<0 or a>90 else a
    base_slope=float(np.degrees(np.arctan(np.tan(np.radians(mline)))))   # line direction -> slope of y vs x, deg
gap=None if (base_slope is None or slope_part is None) else round(base_slope-float(np.degrees(np.arctan(slope_part))),2)
print(json.dumps({'base_line_slope_deg':None if base_slope is None else round(base_slope,2),'yaw_gap_base_minus_part_deg':gap,'tcp_mm':[round(1000*x,1) for x in p],'base_markers':bid,'reproj_px':round(rb,2),'base_yaw_markers_deg':round(yaw_m,2),
  'base_world_mm':[round(1000*x,1) for x in bw],'n_far':len(far),'n_left':len(left),'far_y':[y for _,y in far][:6],'left_x':[x for x,_ in left][:6],
  'part_far_slope_deg':None if slope_part is None else round(float(np.degrees(np.arctan(slope_part))),2),'marker_line_deg':None if mline is None else round(mline,2),
  'd_world_real_minus_target_mm':None if d is None else [round(1000*float(v),1) for v in d],'info':info}))
