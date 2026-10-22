"""Record the held part's far/left outline over bare table (rigid in the image for this grasp), then return over the
aim at the clear height.  python3 outline_table.py OUT.json"""
import sys, subprocess, json, time, numpy as np, cv2
REPO='/home/aldrin/Documents/ros-mcp-server-verified'; S='/tmp/claude-1000/-home-aldrin-Documents-ros-mcp-server-verified/89794742-dc22-4591-870c-0904ba3621f7/scratchpad/'
sys.path[:0]=[REPO, REPO+'/'+'abl'+'ations']
from compliant_insertion_studio.wrapper.marker_yaw import _tcp_pose_now, still_frame, held_outline_px
out=sys.argv[1]; BX,BY=0.106,-0.477; Z_CLEAR=0.229
def mv(x,y,z,q):
    a=[sys.executable,'-u','-m','primitives.move_cartesian','--mode','real','--xyz',f"{x:.5f}",f"{y:.5f}",f"{z:.5f}",'--quat',*[f"{v:.6f}" for v in q],'--min-z','0.15']
    r=subprocess.run(a,cwd=REPO,capture_output=True,text=True,timeout=120); L=r.stdout.splitlines()
    try: return json.loads(L[L.index('__RESULT_JSON__')+1])['result']
    except Exception: return 'failure: '+(L[-1] if L else r.stderr[-120:])
def em(on): subprocess.run(['ros2','param','set','/camera/camera','depth_module.emitter_enabled','1' if on else '0'],capture_output=True,timeout=10); time.sleep(0.8)
p0,q=_tcp_pose_now(); aim=(p0[0],p0[1])
print('up', mv(aim[0],aim[1],0.32,q)); print('to table', mv(BX+0.17,BY,0.32,q))
em(False); img,_=still_frame(n_avg=6); em(True)
o=held_outline_px(img)
if o is None: print('no outline'); cv2.imwrite(S+'table_fail.png',np.clip(img.astype(np.float32)*3,0,255).astype(np.uint8)); sys.exit(1)
far=o['far']; slope=float(np.degrees(np.arctan(np.polyfit([x for x,_ in far],[y for _,y in far],1)[0])))
json.dump({'far':far,'left':o['left'],'far_slope_deg':slope},open(out,'w'))
v=np.clip(img.astype(np.float32)*3,0,255).astype(np.uint8)
for x,y in far+o['left']: cv2.circle(v,(int(x),int(y)),3,(255,0,255),-1)
cv2.imwrite(S+'table_outline.png',v)
print(json.dumps({'n_far':len(far),'n_left':len(o['left']),'far_y_first':[y for _,y in far][:5],'left_x_first':[x for x,_ in o['left']][:5],'far_slope_deg':round(slope,2)}))
print('back over aim', mv(aim[0],aim[1],0.32,q), mv(aim[0],aim[1],Z_CLEAR,q))
