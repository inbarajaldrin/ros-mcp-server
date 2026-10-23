"""One agent step: lift clear (10 mm), turn, move, lower to the measuring height, measure (same_frame), and if
within tolerance descend guarded and report the stop height.  python3 loop_step.py TAG TURN_DEG DX_MM DY_MM [force_descend]"""
import sys, subprocess, json, numpy as np, time
REPO='/home/aldrin/Documents/ros-mcp-server-verified'; S='/tmp/claude-1000/-home-aldrin-Documents-ros-mcp-server-verified/89794742-dc22-4591-870c-0904ba3621f7/scratchpad/'
sys.path[:0]=[REPO, REPO+'/'+'abl'+'ations']
from compliant_insertion_studio.wrapper.marker_yaw import _tcp_pose_now
from compliant_insertion_studio.wrapper.align_at_hover import turn_wrist
tag,turn,dx,dy=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])/1000,float(sys.argv[4])/1000; force=len(sys.argv)>5
Z_CLEAR,Z_MEAS,Z_CONTACT,Z_SEAT=0.2290,0.2216,0.2192,0.1867
def mv(x,y,z,q,guard=None,dur=None):
    a=[sys.executable,'-u','-m','primitives.move_cartesian','--mode','real','--xyz',f"{x:.5f}",f"{y:.5f}",f"{z:.5f}",'--quat',*[f"{v:.6f}" for v in q],'--min-z','0.15']
    if guard: a+=['--max-force',str(guard)]
    if dur: a+=['--min-duration',str(dur)]
    r=subprocess.run(a,cwd=REPO,capture_output=True,text=True,timeout=120); L=r.stdout.splitlines()
    try: return json.loads(L[L.index('__RESULT_JSON__')+1])
    except Exception: return {'result':'failure','error':(L[-1] if L else r.stderr[-200:])}
p,q=_tcp_pose_now(); print('lift', mv(p[0],p[1],Z_CLEAR,q)['result'])
if abs(turn)>0.01: print('turn', turn, turn_wrist(turn)); time.sleep(0.3)
p,q=_tcp_pose_now()
if abs(dx)>1e-5 or abs(dy)>1e-5: print('move', round(1000*dx,1), round(1000*dy,1), mv(p[0]+dx,p[1]+dy,Z_CLEAR,q)['result'])
p,q=_tcp_pose_now(); r=mv(p[0],p[1],Z_MEAS,q,guard=4,dur=2); print('lower to measure', r['result'], r.get('error',''))
m=subprocess.run([sys.executable,S+'same_frame.py',S+f'{tag}.png','26.2'],cwd=REPO,capture_output=True,text=True,timeout=120)
L=[l for l in m.stdout.splitlines() if l.startswith('{')]; res=json.loads(L[-1]) if L else {}
d=res.get('d_world_real_minus_target_mm'); ml=res.get('marker_line_deg'); ps=res.get('part_far_slope_deg')
base_slope=None if ml is None else ((ml+180)%360-180)
if base_slope is not None: base_slope=-((abs(base_slope)-180) if abs(base_slope)>90 else base_slope)   # line direction -> slope deg (y vs x)
print(json.dumps({'tag':tag,'tcp_mm':res.get('tcp_mm'),'d_mm':d,'base_line_slope_deg':None if base_slope is None else round(base_slope,2),'part_edge_slope_deg':ps,'yaw_img_deg':(res.get('info') or {}).get('yaw_img_deg'),'base_yaw_markers':res.get('base_yaw_markers_deg'),'reproj':res.get('reproj_px')}))
yi=(res.get('info') or {}).get('yaw_img_deg'); ok = d is not None and abs(d[0])<=0.8 and abs(d[1])<=0.8 and yi is not None and abs(yi)<=0.4
if ok or force:
    p,q=_tcp_pose_now(); r=mv(p[0],p[1],Z_SEAT-0.002,q,guard=4,dur=6); pz,_=_tcp_pose_now()
    print('DESCENT', r.get('result'), r.get('error',''), 'stopped at z', round(1000*pz[2],1), 'mm =', round(1000*(pz[2]-Z_SEAT),1), 'mm above the seat')
else:
    print('not descending: |d| > 0.8 mm or |yaw_img| > 0.4 deg')
