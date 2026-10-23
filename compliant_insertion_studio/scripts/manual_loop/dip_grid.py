"""Fine dip grid around the current aim: hex lattice PITCH within RMAX, each a guarded dip from 1.5 mm over the rim;
a dip that goes >2 mm below the rim contact continues to the seat.  python3 dip_grid.py PITCH_MM RMAX_MM [YAW_DEG]"""
import sys, subprocess, json, math, numpy as np, time
REPO='/home/aldrin/Documents/ros-mcp-server-verified'; sys.path[:0]=[REPO, REPO+'/'+'abl'+'ations']
from scipy.spatial.transform import Rotation as R
from compliant_insertion_studio.wrapper.marker_yaw import _tcp_pose_now
pitch,rmax=float(sys.argv[1])/1000,float(sys.argv[2])/1000; yaw=float(sys.argv[3]) if len(sys.argv)>3 else 0.0
Z_RIM,Z_SEAT=0.2191,0.1867; Z_START=Z_RIM+0.0015
def mv(x,y,z,q,guard=None,dur=None):
    a=[sys.executable,'-u','-m','primitives.move_cartesian','--mode','real','--xyz',f"{x:.5f}",f"{y:.5f}",f"{z:.5f}",'--quat',*[f"{v:.6f}" for v in q],'--min-z','0.15']
    if guard: a+=['--max-force',str(guard)]
    if dur: a+=['--min-duration',str(dur)]
    r=subprocess.run(a,cwd=REPO,capture_output=True,text=True,timeout=120); L=r.stdout.splitlines()
    try: return json.loads(L[L.index('__RESULT_JSON__')+1])
    except Exception: return {'result':'failure','error':(L[-1] if L else r.stderr[-120:])}
def hex_grid(p,rm):
    n=int(math.ceil(rm/p))+1; pts=[]
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            x,y=p*(i+0.5*j),p*(math.sqrt(3)/2)*j
            if math.hypot(x,y)<=rm+1e-9: pts.append((x,y))
    return sorted(pts,key=lambda v:(round(math.hypot(*v),6),math.atan2(v[1],v[0])))
p,q0=_tcp_pose_now(); ax,ay=p[0],p[1]
q=(R.from_euler('z',yaw,degrees=True)*R.from_quat(q0)).as_quat() if abs(yaw)>0.01 else q0
print('aim', round(1000*ax,1), round(1000*ay,1), 'yaw', yaw); mv(ax,ay,Z_START+0.004,q)
for k,(dx,dy) in enumerate(hex_grid(pitch,rmax)):
    x,y=ax+dx,ay+dy; mv(x,y,Z_START,q)
    r=mv(x,y,Z_RIM-0.006,q,guard=4,dur=2); pz,_=_tcp_pose_now(); z=pz[2]
    tag='rim' if z>Z_RIM-0.002 else 'ENTERED'
    print(f"{k:2d} ({1000*dx:+.1f},{1000*dy:+.1f}) mm -> z {1000*z:.2f} ({1000*(z-Z_RIM):+.2f} vs rim) {tag}", flush=True)
    if tag=='ENTERED':
        r=mv(x,y,Z_SEAT-0.002,q,guard=6,dur=8); pz,_=_tcp_pose_now()
        print(f"   pushed to z {1000*pz[2]:.2f} = {1000*(pz[2]-Z_SEAT):+.2f} mm above the seat; {r.get('error','')}", flush=True)
        sys.exit(0)
    mv(x,y,Z_START,q)
mv(ax,ay,Z_START+0.004,q); print('no entry')
