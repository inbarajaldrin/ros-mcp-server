"""Chain-free pocket-vs-part offset from ONE still frame at the hover: base markers and the part's own marker, each
solved for TRANSLATION ONLY with its rotation taken from the chain (2 base markers / 1 part marker cannot pin rotation
and depth together; with rotation fixed the 8 / 4 corners give translation well). Also reports where the chain's
(servo's) base pose projects vs the detected base markers = the servo target's bias at this height.
python3 marker_check.py OUT.png TCP_Z BX BY BZ BYAW_DEG HQx HQy HQz HQw"""
import sys, subprocess, time, json, numpy as np, cv2
REPO='/home/aldrin/Documents/ros-mcp-server-verified'; sys.path[:0]=[REPO, REPO+'/'+'abl'+'ations']
from scipy.spatial.transform import Rotation as R
from seat_check import X_CAM
from handeye_icp_survey import T as _T
from compliant_insertion_studio.wrapper.marker_yaw import still_frame, _tcp_pose_now, _intrinsics, _layout, _detect
from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
a=sys.argv; out=a[1]; zt=float(a[2]); bp=np.array([float(a[3]),float(a[4]),float(a[5])]); byaw=float(a[6]); hq=[float(x) for x in a[7:11]]
p,q=_tcp_pose_now()
if abs(p[2]-zt)>0.002:
    r=subprocess.run([sys.executable,'-u','-m','primitives.move_cartesian','--mode','real','--xyz',f"{p[0]:.5f}",f"{p[1]:.5f}",f"{zt:.4f}",
        '--quat',*[f"{v:.6f}" for v in q],'--min-z','0.15'],cwd=REPO,capture_output=True,text=True,timeout=90)
def em(on): subprocess.run(['ros2','param','set','/camera/camera','depth_module.emitter_enabled','1' if on else '0'],capture_output=True,timeout=10); time.sleep(0.8)
em(False); img,_=still_frame(n_avg=6); p,q=_tcp_pose_now(); em(True)
K,D=_intrinsics(); Twc=_T(p,q)@X_CAM; Tcw=np.linalg.inv(Twc)
dn,lay_b=_layout('base1'); _,lay_p=_layout('u_brown'); corners,ids=_detect(img,dn)
det={int(i):c.reshape(-1,2) for c,i in zip(corners,ids)} if ids is not None else {}
Rwb=R.from_euler('z',byaw,degrees=True); Rcb=Tcw[:3,:3]@Rwb.as_matrix()                   # base rotation, camera frame (chain)
Rwp=R.from_quat(hq); Rcp=Tcw[:3,:3]@Rwp.as_matrix()                                       # held part rotation, camera frame (chain)
def corners3d(lay,i):
    t,Rm,edge=lay[i]; h=edge/2; return np.array([Rm@np.array(lc)+t for lc in ([-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0])])
def solve_t(Rc0, pts3, pix):
    """translation only, with the in-plane yaw (about the object's own z) searched +-6 deg: (u',v') normalised
    undistorted; u'(r3.X+t3)=r1.X+t1 ...  Returns t, reproj px, yaw correction deg, Rc used."""
    best=None
    for dz in np.arange(-6,6.01,0.1):
        Rc=Rc0@R.from_euler('z',dz,degrees=True).as_matrix()
        t,rp=_solve_t(Rc,pts3,pix)
        if best is None or rp<best[1]: best=(t,rp,dz,Rc)
    return best
def _solve_t(Rc, pts3, pix):
    nrm=cv2.undistortPoints(np.array(pix,np.float32).reshape(-1,1,2),K,D).reshape(-1,2)
    A,b=[],[]
    for X,(u,v) in zip(pts3,nrm):
        Xc=Rc@X
        A.append([1,0,-u]); b.append(u*Xc[2]-Xc[0]); A.append([0,1,-v]); b.append(v*Xc[2]-Xc[1])
    t,res,_,_=np.linalg.lstsq(np.array(A),np.array(b),rcond=None)
    Pc=(Rc@np.array(pts3).T).T+t; uv,_=cv2.projectPoints(Pc.astype(np.float32),np.zeros(3),np.zeros(3),K,D)
    return t, float(np.linalg.norm(uv.reshape(-1,2)-np.array(pix),axis=1).mean())
outj={'tcp_mm':[round(1000*x,1) for x in p],'detected':sorted(det)}
bid=[i for i in det if i in lay_b]; pid=[i for i in det if i in lay_p]
vis=np.clip(img.astype(np.float32)*3,0,255).astype(np.uint8)
if len(bid)>=1:
    pts=np.concatenate([corners3d(lay_b,i) for i in bid]); pix=np.concatenate([det[i] for i in bid])
    tb,rb,dzb,Rcb=solve_t(Rcb,pts,pix)
    # chain: where the servo's base pose puts those corners in this image
    Pw=(Rwb.apply(pts))+bp; Pc=(Tcw[:3,:3]@Pw.T).T+Tcw[:3,3]; uvc,_=cv2.projectPoints(Pc.astype(np.float32),np.zeros(3),np.zeros(3),K,D); uvc=uvc.reshape(-1,2)
    dpx=(pix-uvc); Z=float(Pc[:,2].mean())
    tb_chain=Tcw[:3,:3]@bp+Tcw[:3,3]
    outj['base']={'ids':bid,'t_cam_mm':[round(1000*x,1) for x in tb],'reproj_px':round(rb,2),'yaw_corr_deg':round(float(dzb),2),'chain_t_cam_mm':[round(1000*x,1) for x in tb_chain],
        'chain_minus_markers_cam_mm':[round(1000*x,1) for x in (tb_chain-tb)],'chain_corner_err_px_mean':[round(float(x),1) for x in dpx.mean(0)],'depth_m':round(Z,3)}
    for k in range(len(pix)): cv2.circle(vis,tuple(np.round(uvc[k]).astype(int)),4,(255,0,255),-1)      # chain projection = magenta
    for k in range(len(pix)): cv2.circle(vis,tuple(np.round(pix[k]).astype(int)),3,(0,255,0),-1)        # detected = green
if len(pid)>=1:
    pts=np.concatenate([corners3d(lay_p,i) for i in pid]); pix=np.concatenate([det[i] for i in pid])
    tp,rp,dzp,Rcp=solve_t(Rcp,pts,pix)
    outj['part']={'ids':pid,'t_cam_mm':[round(1000*x,1) for x in tp],'reproj_px':round(rp,2),'yaw_corr_deg':round(float(dzp),2)}
    # chain's held part origin: TCP + chain offset?  report the part origin world xy from markers instead
    pw=Twc[:3,:3]@tp+Twc[:3,3]; outj['part']['origin_world_mm']=[round(1000*x,1) for x in pw]
if len(bid)>=1 and len(pid)>=1:
    sp,sq=load_seat_pose_in_base('base1','u_brown')
    seat_cam=Rcb@sp+tb                                   # where the part origin must be (camera frame), at the seat
    d_cam=seat_cam-tp; d_base=Rcb.T@d_cam                # in base axes: x,y in-plane, z = how far above the seat
    d_world=Rwb.apply(d_base)
    outj['part_to_seat']={'d_base_mm':[round(1000*x,1) for x in d_base],'move_world_xy_mm':[round(1000*x,1) for x in d_world[:2]],'lift_mm':round(1000*d_base[2],1)}
    # yaw check: part marker PnP rotation vs chain
    ok,rv,tv=cv2.solvePnP(pts.astype(np.float32),pix.astype(np.float32),K,D,flags=cv2.SOLVEPNP_IPPE)
    Rpnp=R.from_rotvec(rv.ravel()).as_matrix(); dR=R.from_matrix(Rcp.T@Rpnp).as_euler('xyz',degrees=True)
    outj['part_rot_pnp_minus_chain_deg_partframe']=[round(float(x),2) for x in dR]
print(json.dumps(outj)); cv2.imwrite(out,vis)
