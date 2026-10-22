"""Base-top tilt from guarded touches with the held part at 4 offsets around the aim (world axes); returns to the aim
at Z_CLEAR.  python3 rim_plane.py"""
import sys, subprocess, json, numpy as np
REPO='/home/aldrin/Documents/ros-mcp-server-verified'; sys.path[:0]=[REPO, REPO+'/'+'abl'+'ations']
from compliant_insertion_studio.wrapper.marker_yaw import _tcp_pose_now
Z_CLEAR=0.229
def mv(x,y,z,q,guard=None,dur=None):
    a=[sys.executable,'-u','-m','primitives.move_cartesian','--mode','real','--xyz',f"{x:.5f}",f"{y:.5f}",f"{z:.5f}",'--quat',*[f"{v:.6f}" for v in q],'--min-z','0.15']
    if guard: a+=['--max-force',str(guard)]
    if dur: a+=['--min-duration',str(dur)]
    r=subprocess.run(a,cwd=REPO,capture_output=True,text=True,timeout=120); L=r.stdout.splitlines()
    try: return json.loads(L[L.index('__RESULT_JSON__')+1])
    except Exception: return {'result':'failure','error':(L[-1] if L else r.stderr[-120:])}
p,q=_tcp_pose_now(); ax,ay=p[0],p[1]; print('aim', round(1000*ax,1), round(1000*ay,1)); mv(ax,ay,Z_CLEAR,q)
pts=[]
for dx,dy in ((0.022,0),(-0.022,0),(0,0.012),(0,-0.012),(0,0)):
    mv(ax+dx,ay+dy,Z_CLEAR,q); r=mv(ax+dx,ay+dy,0.216,q,guard=4,dur=3); pz,_=_tcp_pose_now()
    print(f"touch ({1000*dx:+.0f},{1000*dy:+.0f}) mm: z {1000*pz[2]:.2f} {'contact' if r.get('contact') else r.get('result')}")
    pts.append((ax+dx,ay+dy,pz[2])); mv(ax+dx,ay+dy,Z_CLEAR,q)
P=np.array(pts); A=np.c_[P[:,0],P[:,1],np.ones(len(P))]; c=np.linalg.lstsq(A,P[:,2],rcond=None)[0]
n=np.array([-c[0],-c[1],1]); n/=np.linalg.norm(n)
print(json.dumps({'dz_dx':round(float(c[0]),4),'dz_dy':round(float(c[1]),4),'tilt_deg':round(float(np.degrees(np.arccos(n[2]))),2),'normal':[round(float(v),4) for v in n],'resid_mm':[round(1000*float(v),2) for v in (P[:,2]-A@c)]}))
mv(ax,ay,Z_CLEAR,q)
