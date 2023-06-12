import torch
def taylor_A(x,nth=10):
    # Taylor expansion of sin(x)/x
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        if i>0: denom *= (2*i)*(2*i+1)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans
def taylor_B(x,nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans
def taylor_C(x,nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def skew_symmetric(w):
    w0,w1,w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                        torch.stack([w2,O,-w0],dim=-1),
                        torch.stack([-w1,w0,O],dim=-1)],dim=-2)
    return wx

def se3_to_SE3(wu): # [...,3]
    w,u = wu.split([3,3],dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[...,None,None]
    I = torch.eye(3,device=w.device,dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    R = I+A*wx+B*wx@wx
    V = I+B*wx+C*wx@wx
    Rt = torch.cat([R,(V@u[...,None])],dim=-1)
    return Rt

def compose_pair(pose_a, pose_b):
    """
    Inputs:
        pose_a: (N, 3, 4)
        pose_b: (N, 3, 4)
    Outputs:
        pose_b: (N, 4, 4)
    """
    # pose_new(x) = pose_b o pose_a(x)
    R_a, t_a = pose_a[...,:3], pose_a[...,3:]
    R_b, t_b = pose_b[...,:3], pose_b[...,3:]
    R_new = R_b@R_a
    t_new = (R_b@t_a+t_b)[...,0]
    R_new = R_new.float()
    t_new = t_new.float()
    pose_new = torch.cat([R_new,t_new[...,None]],dim=-1) # [..., 3, 4]
    pose_bottom = torch.ones((pose_new.shape[0], 1, 4))
    pose_4x4 = torch.cat([pose_new, pose_bottom], dim = 1)
    return pose_4x4
def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

# basic operations of transforming 3D points between world/camera/image coordinates
def world2cam(X,pose): # [B,N,3]
    X_hom = to_hom(X)
    return X_hom@pose.transpose(-1,-2)

def cam2img(X,cam_intr):
    return X@cam_intr.transpose(-1,-2)

def img2cam(X,cam_intr):
    return X@cam_intr.inverse().transpose(-1,-2)

def cam2world(X,pose):
    # pose is c2w
    X_hom = to_hom(X)
    # pose_inv = invert(pose)
    return X_hom@pose.transpose(-1,-2)