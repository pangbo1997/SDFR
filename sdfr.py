import torch
import os
import numpy as np
from tqdm import tqdm
from lib import rendering
from lib.options import parse_options
from lib.models.OctreeSDF import OctreeSDF
import torch.nn as nn
from pytorch3d.transforms import se3_log_map, so3_log_map,so3_exp_map
from torch.optim import Adam
import open3d as o3d
import json
import trimesh
from pytorch3d.transforms import *

import pytorch3d
from pytorch3d.ops.points_alignment import iterative_closest_point


def get_sdf_net(args,sdfr_model_path,num_lods=5):

    args.num_lods=num_lods
    sdf_net = OctreeSDF(args).cuda()
    sdf_net.load_state_dict(
        torch.load(sdfr_model_path,map_location=torch.device('cuda:{}'.format(args.gpu))))
    sdf_net.eval()
    return sdf_net


def SDFR_lm(sdf_net,trg_pts,obj_scale,init_pose=None):

    trg_pts=pytorch3d.ops.sample_farthest_points(trg_pts, K=150)[0]
    if init_pose==None:
        init_pose=np.diag(np.ones(4))
    trg_pts=trg_pts.to(torch.float32).cuda()
    w = nn.Parameter(so3_log_map(torch.tensor(init_pose[:,:3,:3])).cuda(),requires_grad=True)
    optim_t = nn.Parameter(trg_pts.mean(dim=1),requires_grad=True)

    def cal_delta(J,f):
        J=J.unsqueeze(1)
        return torch.bmm(torch.bmm(J,J.permute(0,2,1))**-1,J).squeeze()*f

    
    for _itr in range(20):

        if w.grad is not None:
            w.grad.zero_()
        if optim_t.grad is not None:
            optim_t.grad.zero_()
            
        optim_R = so3_exp_map(w)
        scene_to_mesh_pts = (trg_pts - optim_t.unsqueeze(1)) @ optim_R *obj_scale*1000
        sdf_value_scene=sdf_net(scene_to_mesh_pts)
        loss = torch.abs(sdf_value_scene).sum()

        loss.backward()
        f=sdf_value_scene.sum(dim=1)
        
        w.data=w.data-cal_delta(w.grad,f)
        optim_t.data=optim_t.data-cal_delta(optim_t.grad,f)

    return so3_exp_map(w).detach().cpu(),optim_t.detach().cpu()


def SDFR_adam(sdf_net,trg_pts,obj_scale,init_pose=None,gt_pose=None,refine_scale=False,initial_scale=None,sample_farthest=False):

    if sample_farthest:
        trg_pts=pytorch3d.ops.sample_farthest_points(trg_pts, K=150)[0]

    if init_pose==None:
        init_pose=np.diag(np.ones(4))

    trg_pts=trg_pts.to(torch.float32).cuda()
    w = nn.Parameter(so3_log_map(torch.tensor(init_pose[:,:3,:3])).cuda(),requires_grad=True)
    optim_t = nn.Parameter(trg_pts.mean(dim=1),requires_grad=True)


    optimizer_R = Adam([w], lr=5e-2)
    optimizer_t = Adam([optim_t], lr=3e-3)
    if refine_scale:

        optim_z = nn.Parameter(initial_scale[:,None,None],requires_grad=True)
        optimizer_z=Adam([optim_z], lr=3e-2)

    for _itr in range(100):
        optimizer_t.zero_grad()
        optimizer_R.zero_grad()
        if refine_scale:
            optimizer_z.zero_grad()

        optim_R = so3_exp_map(w)
        scene_to_mesh_pts = ((trg_pts - optim_t.unsqueeze(1))) @ optim_R *obj_scale*1000
        if refine_scale:
            scene_to_mesh_pts/=optim_z
        sdf_value_scene=sdf_net(scene_to_mesh_pts)
        loss = torch.abs(sdf_value_scene).mean()

        loss.backward()
        optimizer_R.step()
        optimizer_t.step()

        if refine_scale:
            optimizer_z.step()

    if refine_scale:
        return so3_exp_map(w).detach().cpu(),optim_t.detach().cpu(),optim_z.detach().cpu()
    else:
        return so3_exp_map(w).detach().cpu(),optim_t.detach().cpu(),None
