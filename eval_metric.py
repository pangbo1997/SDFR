import os
import glob
import json
import numpy as np
import torch
from inout import load_ply
import os

from scipy.spatial.transform import Rotation 
import numpy as np
import json
from tqdm import tqdm
from sklearn.neighbors import KDTree
import torch
from lib import rendering
import glob
from sdfr import *
from pytorch3d.ops.points_alignment import iterative_closest_point
from sdfr import get_sdf_net
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer,
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    Textures,
    TexturesVertex,PerspectiveCameras
)
from pytorch3d.renderer import *
from pytorch3d.utils import cameras_from_opencv_projection
import matplotlib.pyplot as plt
import cv2
import misc, visibility
import time
from scipy import spatial
import warnings



def cal_5cm_5deg_error(pred_pose,gt_pose):

    eps=1e-6
    R,t=pred_pose[:,:3,:3],pred_pose[:,:3,3]
    r_loss=(torch.einsum('bii->b',torch.bmm(R,gt_pose[:,:3,:3].permute(0,2,1)))-1)/2
    r_loss=torch.clip(r_loss,min=-1+eps,max=1-eps)
    R_error=torch.arccos(r_loss)*180/np.pi
    t_error=(t-gt_pose[:,:3,3]).norm(dim=1)
    return R_error,t_error


def render(extrinsic,obj_mesh):
    image_size=torch.tensor([480,640]).reshape(1,2)
    device=torch.device('cuda')
    query_camera=np.array([[1598.5735940340862/3,0,  954.56/3],[0, 1598.5735940340862/3, 716.6687979261368/3],[0,         0,         1]])
    

    R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
    cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
    raster_settings = RasterizationSettings(image_size=(480,640), blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(obj_mesh.to(device))
    depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
    depth[depth==-1]=0
    return depth

def cal_vsd_error(pred_pose, gt_pose, mesh_model, delta=1, tau=0.02, cost_type='step'):
    """
    Visible Surface Discrepancy.

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :param depth_test: Depth image of the test scene.
    :param K: Camera matrix.
    :param delta: Tolerance used for estimation of the visibility masks.
    :param tau: Misalignment tolerance.
    :param cost_type: Pixel-wise matching cost:
        'tlinear' - Used in the original definition of VSD in:
            Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW 2016
        'step' - Used for SIXD Challenge 2017. It is easier to interpret.
    :return: Error of pose_est w.r.t. pose_gt.
    """

    K=np.array([[1598.5735940340862/3,0,  954.56/3],[0, 1598.5735940340862/3, 716.6687979261368/3],[0,         0,         1]])

    im_size = (640, 480)

    depth_est = render(pred_pose,mesh_model)


    depth_gt = render(gt_pose,mesh_model)

    # import pdb;pdb.set_trace()
    depth_test=depth_gt.copy()

    # Convert depth images to distance images
    dist_test = misc.depth_im_to_dist_im(depth_test, K)
    dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
    dist_est = misc.depth_im_to_dist_im(depth_est, K)
    # Visibility mask of the model in the ground truth pose
    visib_gt = visibility.estimate_visib_mask_gt(dist_test, dist_gt, delta)

    # Visibility mask of the model in the estimated pose
    visib_est = visibility.estimate_visib_mask_est(dist_test, dist_est, visib_gt, delta)

    # Intersection and union of the visibility masks
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    # Pixel-wise matching cost
    costs = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    if cost_type == 'step':
        costs = costs >= tau
    elif cost_type == 'tlinear': # Truncated linear
        costs *= (1.0 / tau)
        costs[costs > 1.0] = 1.0
    else:
        print('Error: Unknown pixel matching cost.')
        exit(-1)

    # costs_vis = np.ones(dist_gt.shape)
    # costs_vis[visib_inter] = costs
    # import matplotlib.pyplot as plt
    # plt.matshow(costs_vis)
    # plt.colorbar()
    # plt.show()

    # Visible Surface Discrepancy
    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()
    if visib_union_count > 0:
        e = (costs.sum() + visib_comp_count) / float(visib_union_count)
    else:
        e = 1.0
    return e


def cal_add_error(pred_pose,gt_pose,model):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    # import pdb;pdb.set_trace()
    pts_est = misc.transform_pts_Rt(model, pred_pose[:3,:3], pred_pose[:3,3])
    pts_gt = misc.transform_pts_Rt(model,gt_pose[:3,:3], gt_pose[:3,3])
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return     e

def cal_adds_error(pred_pose,gt_pose,model):
    """
    Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = misc.transform_pts_Rt(model, pred_pose[:3,:3], pred_pose[:3,3])
    pts_gt = misc.transform_pts_Rt(model,gt_pose[:3,:3], gt_pose[:3,3])

    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e

def eval_5cm_5deg(dataset_dir,result_dir,eval_method,mode,render=False):
    # import pdb;pdb.set_trace()
    result_dir=os.path.join(result_dir,eval_method,mode)
    final_result_json={}
    if render:
        isrender='render'
        _iter=glob.glob(os.path.join(result_dir,'*/*_render.json'))
    else:
        if eval_method=='sdfr':
            isrender=''
            _iter=glob.glob(os.path.join(result_dir,'*/*.json'))
        else:
            isrender='norender'
            _iter=glob.glob(os.path.join(result_dir,'*/*_norender.json'))

    if os.path.exists(f'results_metric/5cm5deg/result_{eval_method}_{mode}{isrender}5cm5deg.json'): return 

    for obj_json in tqdm(_iter):
        obj_record=json.load(open(obj_json))
        obj_name=obj_json.split('/')[-2]
        dataset_name=obj_name.split('_')[0]
        if render:
            test_exp_name=obj_json.split('/')[-1][:-12]
        else:
            if eval_method=='sdfr':
                test_exp_name=obj_json.split('/')[-1][:-5]
            else:
                test_exp_name=obj_json.split('/')[-1][:-14]

        R_error_list=[]
        t_error_list=[]
        # trg_ply_file=glob.glob(os.path.join(dataset_dir,mode,obj_name,test_exp_name,'target/*.ply'))
        gt_file_list=json.load(open(os.path.join(dataset_dir,mode,obj_name,test_exp_name,'gt_file_list.json')))
        # import pdb;pdb.set_trace()
        for i,pred_pose in enumerate(obj_record['pred_pose_list']):
            # trg_ply=trg_ply_file[i]
            gt_pose=np.loadtxt(gt_file_list[i])
            # gt_pose=np.loadtxt(trg_ply.replace('target','gt_pose').replace('.ply','.txt').replace('model_',''))#np.loadtxt(os.path.join(dataset_dir,'standard',dataset_name,test_exp_name,'gt_pose/{}.txt'.format(i)))
            R_error,t_error=cal_5cm_5deg_error(torch.tensor(pred_pose,dtype=torch.float32)[None],torch.tensor(gt_pose,dtype=torch.float32)[None])
            R_error_list.append(R_error[0].detach().cpu().numpy())
            t_error_list.append(t_error[0].detach().cpu().numpy())
            
        metric_5cm_5deg=(((np.array(R_error_list)<5) & (np.array(t_error_list)<0.05)).sum())/len(obj_record['pred_pose_list']) 
        # import pdb;pdb.set_trace()
        if dataset_name not in final_result_json.keys():
            final_result_json[dataset_name]={}
        if test_exp_name not in final_result_json[dataset_name].keys():
            final_result_json[dataset_name][test_exp_name]=[]
        final_result_json[dataset_name][test_exp_name].append(metric_5cm_5deg)
    for dataset_name in final_result_json.keys():
        for test_exp_name in final_result_json[dataset_name].keys():
            final_result_json[dataset_name][test_exp_name]=np.mean(final_result_json[dataset_name][test_exp_name])
    with open(f'results_metric/5cm5deg/result_{eval_method}_{mode}{isrender}5cm5deg.json','w') as f:
        json.dump(final_result_json,f,indent=2)
        
def eval_vsd(dataset_dir,result_dir,eval_method,mode,render=False):
    # import pdb;pdb.set_trace()
    result_dir=os.path.join(result_dir,eval_method,mode)
    final_result_json={}
    if render and eval_method!='sdfr':
        isrender='render'
        _iter=glob.glob(os.path.join(result_dir,'*/*_render.json'))
    else:
        if eval_method=='sdfr':
            isrender=''
            _iter=glob.glob(os.path.join(result_dir,'*/*.json'))
        else:
            isrender='norender'
            _iter=glob.glob(os.path.join(result_dir,'*/*_norender.json'))

    if os.path.exists(f'results_metric/vsd/result_{eval_method}_{mode}{isrender}vsd.json'): return 

    for obj_json in tqdm(_iter):
        if mode=='standard' or mode=='noise' or mode=='scale':
            if not 'tless' in obj_json: continue
        
        obj_record=json.load(open(obj_json))
        obj_name=obj_json.split('/')[-2]
        dataset_name=obj_name.split('_')[0]
        if render and eval_method!='sdfr':
            test_exp_name=obj_json.split('/')[-1][:-12]
        else:
            if eval_method=='sdfr':
                test_exp_name=obj_json.split('/')[-1][:-5]
            else:
                test_exp_name=obj_json.split('/')[-1][:-14]

        R_error_list=[]
        t_error_list=[]
        gt_file_list=json.load(open(os.path.join(dataset_dir,mode,obj_name,test_exp_name,'gt_file_list.json')))
       
        model_path=os.path.join(dataset_dir.replace('render','models'),obj_name.split('_')[0],'_'.join(obj_name.split('_')[1:])+'.obj')
        mesh_model=load_objs_as_meshes([model_path], device=torch.device('cuda'))
        mesh_model.scale_verts_(0.001)

        vsd_list=[]
        for i,pred_pose in (enumerate(obj_record['pred_pose_list'])):
            gt_pose=np.loadtxt(gt_file_list[i])
            pred_pose,gt_pose=np.array(pred_pose),np.array(gt_pose)
            vsd=cal_vsd_error(pred_pose, gt_pose[:3], mesh_model)
            vsd_list.append(vsd)
        metric_vsd=(np.array(vsd_list)<0.3).sum()/len(obj_record['pred_pose_list']) 


        if dataset_name not in final_result_json.keys():
            final_result_json[dataset_name]={}
        if test_exp_name not in final_result_json[dataset_name].keys():
            final_result_json[dataset_name][test_exp_name]=[]
        final_result_json[dataset_name][test_exp_name].append(metric_vsd)
    for dataset_name in final_result_json.keys():
        for test_exp_name in final_result_json[dataset_name].keys():
            final_result_json[dataset_name][test_exp_name]=np.mean(final_result_json[dataset_name][test_exp_name])
    with open(f'results_metric/vsd/result_{eval_method}_{mode}{isrender}vsd.json','w') as f:
        json.dump(final_result_json,f,indent=2)

def eval_add(dataset_dir,result_dir,eval_method,mode,render=False):
    # import pdb;pdb.set_trace()
    result_dir=os.path.join(result_dir,eval_method,mode)
    final_result_json={}
    if render:
        isrender='render'
        _iter=glob.glob(os.path.join(result_dir,'*/*_render.json'))
    else:
        if eval_method=='sdfr':
            isrender=''
            _iter=glob.glob(os.path.join(result_dir,'*/*.json'))
        else:
            isrender='norender'
            _iter=glob.glob(os.path.join(result_dir,'*/*_norender.json'))

    if os.path.exists(f'results_metric/add/result_{eval_method}_{mode}{isrender}add.json'): return 
    # print('Begin')
    for obj_json in tqdm(_iter):


        obj_record=json.load(open(obj_json))
        obj_name=obj_json.split('/')[-2]
        dataset_name=obj_name.split('_')[0]
        if render:
            test_exp_name=obj_json.split('/')[-1][:-12]
        else:
            if eval_method=='sdfr':
                test_exp_name=obj_json.split('/')[-1][:-5]
            else:
                test_exp_name=obj_json.split('/')[-1][:-14]

        R_error_list=[]
        t_error_list=[]
        gt_file_list=json.load(open(os.path.join(dataset_dir,mode,obj_name,test_exp_name,'gt_file_list.json')))
       
        model_path=os.path.join(dataset_dir.replace('render','models'),obj_name.split('_')[0],'_'.join(obj_name.split('_')[1:])+'.obj')
        mesh_model=load_objs_as_meshes([model_path], device=torch.device('cuda'))
        mesh_model.scale_verts_(0.001)
        diameter_json=json.load(open('/'.join(model_path.split('/')[:-1])+'/models_info.json'))
        obj_id=int(model_path.split('/')[-1][:-4].split('_')[-1])
        diameter=diameter_json[str(obj_id)]['diameter']/1000
        
        add_list=[]
        R_error_list=[]
        t_error_list=[]
        # print(obj_json)
        for i,pred_pose in (enumerate(obj_record['pred_pose_list'])):
            gt_pose=np.loadtxt(gt_file_list[i])
            pred_pose,gt_pose=np.array(pred_pose),np.array(gt_pose)
            add=cal_add_error(pred_pose, gt_pose[:3], mesh_model.verts_list()[0].detach().cpu().numpy())
            add_list.append(add)
            # R_error,t_error=cal_5cm_5deg_error(torch.tensor(pred_pose,dtype=torch.float32)[None],torch.tensor(gt_pose,dtype=torch.float32)[None])
            # R_error_list.append(R_error[0].detach().cpu().numpy())
            # t_error_list.append(t_error[0].detach().cpu().numpy())

        metric_add=(np.array(add_list)<0.1*diameter).sum()/len(obj_record['pred_pose_list']) 
        # metric_5cm_5deg=(((np.array(R_error_list)<5) & (np.array(t_error_list)<0.05)).sum())/len(obj_record['pred_pose_list']) 
        # import pdb;pdb.set_trace()

        if dataset_name not in final_result_json.keys():
            final_result_json[dataset_name]={}
        if test_exp_name not in final_result_json[dataset_name].keys():
            final_result_json[dataset_name][test_exp_name]=[]
        final_result_json[dataset_name][test_exp_name].append(metric_add)
    for dataset_name in final_result_json.keys():
        for test_exp_name in final_result_json[dataset_name].keys():
            final_result_json[dataset_name][test_exp_name]=np.mean(final_result_json[dataset_name][test_exp_name])
    with open(f'results_metric/add/result_{eval_method}_{mode}{isrender}add.json','w') as f:
        json.dump(final_result_json,f,indent=2)

def eval_adds(dataset_dir,result_dir,eval_method,mode,render=False):
    # import pdb;pdb.set_trace()
    result_dir=os.path.join(result_dir,eval_method,mode)
    final_result_json={}
    if render:
        isrender='render'
        _iter=glob.glob(os.path.join(result_dir,'*/*_render.json'))
    else:
        if eval_method=='sdfr':
            isrender=''
            _iter=glob.glob(os.path.join(result_dir,'*/*.json'))
        else:
            isrender='norender'
            _iter=glob.glob(os.path.join(result_dir,'*/*_norender.json'))

    if os.path.exists(f'results_metric/adds/result_{eval_method}_{mode}{isrender}adds.json'): return 

    for obj_json in tqdm(_iter):
        # if not 'lm' in obj_json and not 'ycbv' in obj_json: continue

        obj_record=json.load(open(obj_json))
        obj_name=obj_json.split('/')[-2]
        dataset_name=obj_name.split('_')[0]
        if render:
            test_exp_name=obj_json.split('/')[-1][:-12]
        else:
            if eval_method=='sdfr':
                test_exp_name=obj_json.split('/')[-1][:-5]
            else:
                test_exp_name=obj_json.split('/')[-1][:-14]

        R_error_list=[]
        t_error_list=[]
        gt_file_list=json.load(open(os.path.join(dataset_dir,mode,obj_name,test_exp_name,'gt_file_list.json')))
       
        model_path=os.path.join(dataset_dir.replace('render','models'),obj_name.split('_')[0],'_'.join(obj_name.split('_')[1:])+'.obj')
        mesh_model=load_objs_as_meshes([model_path], device=torch.device('cuda'))
        mesh_model.scale_verts_(0.001)
        diameter_json=json.load(open('/'.join(model_path.split('/')[:-1])+'/models_info.json'))
        obj_id=int(model_path.split('/')[-1][:-4].split('_')[-1])
        diameter=diameter_json[str(obj_id)]['diameter']/1000
        
        adds_list=[]

        for i,pred_pose in (enumerate(obj_record['pred_pose_list'])):
            gt_pose=np.loadtxt(gt_file_list[i])
            pred_pose,gt_pose=np.array(pred_pose),np.array(gt_pose)
            adds=cal_adds_error(pred_pose, gt_pose[:3], mesh_model.verts_list()[0].detach().cpu().numpy())
            adds_list.append(adds)
            # R_error,t_error=cal_5cm_5deg_error(torch.tensor(pred_pose,dtype=torch.float32)[None],torch.tensor(gt_pose,dtype=torch.float32)[None])
            # R_error_list.append(R_error[0].detach().cpu().numpy())
            # t_error_list.append(t_error[0].detach().cpu().numpy())

        metric_adds=(np.array(adds_list)<0.1*diameter).sum()/len(obj_record['pred_pose_list']) 
        # metric_5cm_5deg=(((np.array(R_error_list)<5) & (np.array(t_error_list)<0.05)).sum())/len(obj_record['pred_pose_list']) 
        # import pdb;pdb.set_trace()

        if dataset_name not in final_result_json.keys():
            final_result_json[dataset_name]={}
        if test_exp_name not in final_result_json[dataset_name].keys():
            final_result_json[dataset_name][test_exp_name]=[]
        final_result_json[dataset_name][test_exp_name].append(metric_adds)
    for dataset_name in final_result_json.keys():
        for test_exp_name in final_result_json[dataset_name].keys():
            final_result_json[dataset_name][test_exp_name]=np.mean(final_result_json[dataset_name][test_exp_name])
    with open(f'results_metric/adds/result_{eval_method}_{mode}{isrender}adds.json','w') as f:
        json.dump(final_result_json,f,indent=2)


import multiprocessing
import time


if __name__=='__main__':
    dataset_dir='./render'
    result_dir='./results/'
    os.makedirs('results_metric/5cm5deg',exist_ok=True)
    os.makedirs('results_metric/add',exist_ok=True)
    os.makedirs('results_metric/adds',exist_ok=True)
    os.makedirs('results_metric/vsd',exist_ok=True)

    args_list=[]
    for mode in ['standard','noise','scale','diverse','corrupt']:
        for method in ['symmicp','sdfr','fricp_3']:
            if mode=='standard' or mode=='noise' or mode=='scale':
                eval_5cm_5deg(dataset_dir,result_dir,method,mode,True)
                # eval_add(dataset_dir,result_dir,method,mode,True)
                # eval_vsd(dataset_dir,result_dir,method,mode,True)
                if method!='sdfr':
                    eval_5cm_5deg(dataset_dir,result_dir,method,mode,False)
                    # eval_add(dataset_dir,result_dir,method,mode,False)
                    # eval_vsd(dataset_dir,result_dir,method,mode,False)
            else:
                eval_5cm_5deg(dataset_dir,result_dir,method,mode,True)
                # eval_add(dataset_dir,result_dir,method,mode,True)
                # eval_adds(dataset_dir,result_dir,method,mode,True)
                # eval_vsd(dataset_dir,result_dir,method,mode,True)


            


