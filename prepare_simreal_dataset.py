import open3d as o3d
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
zebra_dataset_path='zebra_dataset'


def write_ply(path,pcd):
    with open(path,'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(pcd.points)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float nx\n')
        f.write('property float ny\n')
        f.write('property float nz\n')
        f.write('end_header\n')
        for pts,normal in zip(np.array(pcd.points),np.array(pcd.normals)):
            f.write('{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(pts[0],pts[1],pts[2],normal[0],normal[1],normal[2]))


def generate_standard_render(root_dir,model_path,num,rot_interval,trans_interval):

    render_path=os.path.join(root_dir,'render/standard','_'.join(model_path.split('/')[-2:])[:-4])

    os.makedirs(render_path,exist_ok=True)

    source_dir='{}/R_{}_{}_t_{}_{}/source'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    target_dir='{}/R_{}_{}_t_{}_{}/target'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    init_pose_dir='{}/R_{}_{}_t_{}_{}/init_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    gt_pose_dir='{}/R_{}_{}_t_{}_{}/gt_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])

    if os.path.exists('{}/model_{}.ply'.format(source_dir,num-1)): return
    obj_mesh=load_objs_as_meshes([model_path], device=torch.device('cuda'))
    dataset_name,obj_name=model_path.split('/')[-2:]
    obj_name=obj_name[:-4]

    if not os.path.exists(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color','0.png')): return 

    os.makedirs(source_dir,exist_ok=True)
    os.makedirs(target_dir,exist_ok=True)
    os.makedirs(init_pose_dir,exist_ok=True)
    os.makedirs(gt_pose_dir,exist_ok=True)





    image_size=torch.tensor(cv2.imread(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color','0.png')).shape[:2]).reshape(1,2)
    device=torch.device('cuda')
    obj_mesh.scale_verts_(0.001)

    # num=len(os.listdir(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color')))
    instance_no=[int(i.split('.')[0]) for i in os.listdir(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color'))]
    for j in tqdm(instance_no):

        matrix=np.loadtxt(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'gt_poses/{j}.txt'))
        matrix[:3,3]*=0.001
        matrix[3,3]=1

        trg_pcd=o3d.io.read_point_cloud(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'trg_pcds/{j}.ply'))
        trg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        # import pdb;pdb.set_trace()
        cltrg_pcd, _  = trg_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        if len(cltrg_pcd.points)==0:
            write_ply('{}/model_{}.ply'.format(target_dir,j),trg_pcd)
        else:
            write_ply('{}/model_{}.ply'.format(target_dir,j),cltrg_pcd)

        # write_ply('{}/model_{}.ply'.format(target_dir,j),trg_pcd)

        query_camera=np.loadtxt(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'cam_K/{j}.txt'))
        

        src_pcd=o3d.geometry.PointCloud()
        while len(src_pcd.points)==0:
            axis=np.random.rand(3)
            axis=axis/np.linalg.norm(axis)
            angle=(np.random.rand()*(rot_interval[1]-rot_interval[0])+rot_interval[0])/180*np.pi
            rotvec=axis*angle
            rot=Rotation.from_rotvec(rotvec).as_matrix()
            trans=np.zeros(3)
            trans[2]=(np.random.rand()*(trans_interval[1]-trans_interval[0])+trans_interval[0])
            trans[1]=(np.random.rand()*0.02)
            trans[0]=(np.random.rand()*0.02)
            if np.random.rand()>0.5:
                trans*=-1  
            delta_matrix=np.zeros((4,4))
            delta_matrix[:3,:3]=rot@matrix[:3,:3]
            delta_matrix[:3,3]=matrix[:3,3]+trans
            delta_matrix[3,3]=1

            extrinsic=delta_matrix#delta_matrix@matrix
            R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
            cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
            raster_settings = RasterizationSettings(image_size=(int(image_size[0][0]),int(image_size[0][1])), blur_radius=0.0, faces_per_pixel=1)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(obj_mesh.to(device))
            depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
            depth[depth==-1]=0
            depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)
            src_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
            src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
            write_ply('{}/model_render_{}.ply'.format(source_dir,j),src_pcd)

        np.savetxt('{}/{}.txt'.format(gt_pose_dir,j),matrix)
        np.savetxt('{}/{}.txt'.format(init_pose_dir,j),extrinsic)
    
def generate_noise_render(root_dir,model_path,num,rot_interval,trans_interval):

    render_path=os.path.join(root_dir,'render/noise','_'.join(model_path.split('/')[-2:])[:-4])
    os.makedirs(render_path,exist_ok=True)

    source_dir='{}/R_{}_{}_t_{}_{}_noise/source'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    target_dir='{}/R_{}_{}_t_{}_{}_noise/target'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    init_pose_dir='{}/R_{}_{}_t_{}_{}_noise/init_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    gt_pose_dir='{}/R_{}_{}_t_{}_{}_noise/gt_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])


    if os.path.exists('{}/model_{}.ply'.format(source_dir,num-1)): return
    obj_mesh=load_objs_as_meshes([model_path], device=torch.device('cuda'))
    dataset_name,obj_name=model_path.split('/')[-2:]
    obj_name=obj_name[:-4]

    
    if os.path.exists('{}/model_{}.ply'.format(source_dir,num-1)): return

    os.makedirs(source_dir,exist_ok=True)
    os.makedirs(target_dir,exist_ok=True)
    os.makedirs(init_pose_dir,exist_ok=True)
    os.makedirs(gt_pose_dir,exist_ok=True)

    image_size=torch.tensor(cv2.imread(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color','0.png')).shape[:2]).reshape(1,2)
    device=torch.device('cuda')
    obj_mesh.scale_verts_(0.001)

    instance_no=[int(i.split('.')[0]) for i in os.listdir(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color'))]
    for j in tqdm(instance_no):
        matrix=np.loadtxt(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'gt_poses/{j}.txt'))
        matrix[:3,3]*=0.001
        matrix[3,3]=1


        trg_pcd=o3d.io.read_point_cloud(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'trg_pcds/{j}.ply'))
        trg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))

        # cltrg_pcd, _  = trg_pcd.remove_radius_outlier(nb_points=16, radius=0.01)
        # if len(cltrg_pcd.points)>=40:
        #     trg_pcd=cltrg_pcd
        # import pdb;pdb.set_trace()
        try:
            pts_array=np.array(trg_pcd.points)
            v_min,v_max=pts_array.min(axis=0),pts_array.max(axis=0)
            kdt = KDTree(pts_array, leaf_size=30, metric='euclidean')
            distance,ind=kdt.query(pts_array, k=6, return_distance=True)
            distance=np.median(distance,axis=1)
            normals=np.array(trg_pcd.normals)
            gauss_noise=normals*((distance)[:,None].repeat(3,axis=1))

            outlier_ratio=np.random.rand()*0.5+0.5
            outlier=np.random.rand(int(outlier_ratio*pts_array.shape[0]),3)
            outlier=outlier*(v_max-v_min)+v_min
            pts_array+=gauss_noise
            pts_array=np.concatenate((pts_array,outlier),axis=0)
            trg_pcd.points=o3d.utility.Vector3dVector(pts_array)
            trg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        except:
            pass
        write_ply('{}/model_{}.ply'.format(target_dir,j),trg_pcd)

        query_camera=np.loadtxt(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'cam_K/{j}.txt'))

        src_pcd=o3d.geometry.PointCloud()
        while len(src_pcd.points)==0:
            axis=np.random.rand(3)
            axis=axis/np.linalg.norm(axis)
            angle=(np.random.rand()*(rot_interval[1]-rot_interval[0])+rot_interval[0])/180*np.pi
            rotvec=axis*angle
            rot=Rotation.from_rotvec(rotvec).as_matrix()
            trans=np.zeros(3)
            trans[2]=(np.random.rand()*(trans_interval[1]-trans_interval[0])+trans_interval[0])
            trans[1]=(np.random.rand()*0.02)
            trans[0]=(np.random.rand()*0.02)
            if np.random.rand()>0.5:
                trans*=-1  
            delta_matrix=np.zeros((4,4))
            delta_matrix[:3,:3]=rot@matrix[:3,:3]
            delta_matrix[:3,3]=matrix[:3,3]+trans
            delta_matrix[3,3]=1

            extrinsic=delta_matrix#delta_matrix@matrix
            R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
            cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
            raster_settings = RasterizationSettings(image_size=(int(image_size[0][0]),int(image_size[0][1])), blur_radius=0.0, faces_per_pixel=1)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(obj_mesh.to(device))
            depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
            depth[depth==-1]=0
            depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)
            # import pdb;pdb.set_trace()
            src_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
            src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
            write_ply('{}/model_render_{}.ply'.format(source_dir,j),src_pcd)

            np.savetxt('{}/{}.txt'.format(gt_pose_dir,j),matrix)
            np.savetxt('{}/{}.txt'.format(init_pose_dir,j),extrinsic)

def generate_scale_render(root_dir,model_path,num,rot_interval,trans_interval,scale_interval):
    
    render_path=os.path.join(root_dir,'render/scale','_'.join(model_path.split('/')[-2:])[:-4])
    os.makedirs(render_path,exist_ok=True)

    source_dir='{}/R_{}_{}_t_{}_{}_scale_{}_{}/source'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1],scale_interval[0],scale_interval[1])
    target_dir='{}/R_{}_{}_t_{}_{}_scale_{}_{}/target'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1],scale_interval[0],scale_interval[1])
    init_pose_dir='{}/R_{}_{}_t_{}_{}_scale_{}_{}/init_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1],scale_interval[0],scale_interval[1])
    gt_pose_dir='{}/R_{}_{}_t_{}_{}_scale_{}_{}/gt_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1],scale_interval[0],scale_interval[1])

    if os.path.exists('{}/model_{}.ply'.format(source_dir,num-1)): return
    obj_mesh=load_objs_as_meshes([model_path], device=torch.device('cuda'))
    dataset_name,obj_name=model_path.split('/')[-2:]
    obj_name=obj_name[:-4]

    if not os.path.exists(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color','0.png')): return 

    os.makedirs(source_dir,exist_ok=True)
    os.makedirs(target_dir,exist_ok=True)
    os.makedirs(init_pose_dir,exist_ok=True)
    os.makedirs(gt_pose_dir,exist_ok=True)


    image_size=torch.tensor(cv2.imread(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color','0.png')).shape[:2]).reshape(1,2)
    device=torch.device('cuda')
    obj_mesh.scale_verts_(0.001)

    instance_no=[int(i.split('.')[0]) for i in os.listdir(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color'))]
    for j in tqdm(instance_no):
        matrix=np.loadtxt(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'gt_poses/{j}.txt'))
        matrix[:3,3]*=0.001
        matrix[3,3]=1

        if np.random.rand()>0.5:
            scale_interval=[0.5,0.8]
        else:
            scale_interval=[1.2,1.5]
        scale=(np.random.rand()*(scale_interval[1]-scale_interval[0])+scale_interval[0])


        trg_pcd=o3d.geometry.PointCloud()
        pts_array=np.array(o3d.io.read_point_cloud(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'trg_pcds/{j}.ply')).points)
        pts_array=(pts_array-matrix[:3,3])*scale+matrix[:3,3]
        trg_pcd.points=o3d.utility.Vector3dVector(pts_array)
        trg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_{}.ply'.format(target_dir,j),trg_pcd)

        src_pcd=o3d.geometry.PointCloud()
        query_camera=np.loadtxt(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'cam_K/{j}.txt'))
        while len(src_pcd.points)==0:
            axis=np.random.rand(3)
            axis=axis/np.linalg.norm(axis)
            angle=(np.random.rand()*(rot_interval[1]-rot_interval[0])+rot_interval[0])/180*np.pi
            rotvec=axis*angle
            rot=Rotation.from_rotvec(rotvec).as_matrix()
            trans=np.zeros(3)
            trans[2]=(np.random.rand()*(trans_interval[1]-trans_interval[0])+trans_interval[0])
            trans[1]=(np.random.rand()*0.02)
            trans[0]=(np.random.rand()*0.02)
            if np.random.rand()>0.5:
                trans*=-1  
            
            # print(scale)
            delta_matrix=np.zeros((4,4))
            delta_matrix[:3,:3]=rot@matrix[:3,:3]
            delta_matrix[:3,3]=matrix[:3,3]+trans
            delta_matrix[3,3]=1
    
            extrinsic=delta_matrix#delta_matrix@matrix
            R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
            cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
            raster_settings = RasterizationSettings(image_size=(int(image_size[0][0]),int(image_size[0][1])), blur_radius=0.0, faces_per_pixel=1)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(obj_mesh.to(device))
            depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
            depth[depth==-1]=0
            depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)

            src_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
            src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
            write_ply('{}/model_render_{}.ply'.format(source_dir,j),src_pcd)

            np.savetxt('{}/{}.txt'.format(gt_pose_dir,j),matrix)
            np.savetxt('{}/{}.txt'.format(init_pose_dir,j),extrinsic)

import math

def generate_real_render(root_dir,model_path,num):

    render_path=os.path.join(root_dir,'render/real','_'.join(model_path.split('/')[-2:])[:-4])
    os.makedirs(render_path,exist_ok=True)
    source_dir='{}/real/source'.format(render_path)
    target_dir='{}/real/target'.format(render_path)
    init_pose_dir='{}/real/init_pose'.format(render_path)
    gt_pose_dir='{}/real/gt_pose'.format(render_path)

    if os.path.exists('{}/model_{}.ply'.format(source_dir,num-1)): return
    obj_mesh=load_objs_as_meshes([model_path], device=torch.device('cuda'))
    dataset_name,obj_name=model_path.split('/')[-2:]
    obj_name=obj_name[:-4]

    if not os.path.exists(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color','0.png')): return 

    os.makedirs(source_dir,exist_ok=True)
    os.makedirs(target_dir,exist_ok=True)
    os.makedirs(init_pose_dir,exist_ok=True)
    os.makedirs(gt_pose_dir,exist_ok=True)





    image_size=torch.tensor(cv2.imread(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color','0.png')).shape[:2]).reshape(1,2)
    device=torch.device('cuda')
    obj_mesh.scale_verts_(0.001)

    instance_no=[int(i.split('.')[0]) for i in os.listdir(os.path.join(zebra_dataset_path,dataset_name,obj_name,'color'))]
    for j in tqdm(instance_no):

        matrix=np.loadtxt(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'gt_poses/{j}.txt'))
        matrix[:3,3]*=0.001
        matrix[3,3]=1

        trg_pcd=o3d.io.read_point_cloud(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'trg_pcds/{j}.ply'))

        # if len(trg_pcd.points)==0: continue

        trg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_{}.ply'.format(target_dir,j),trg_pcd)

        # cltrg_pcd, _  = trg_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        # if len(cltrg_pcd.points)==0:
        #     write_ply('{}/model_{}.ply'.format(target_dir,j),trg_pcd)
        # else:
        #     write_ply('{}/model_{}.ply'.format(target_dir,j),cltrg_pcd)


        # import pdb;pdb.set_trace()


        query_camera=np.loadtxt(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'cam_K/{j}.txt'))
        
        delta_matrix=np.loadtxt(os.path.join(zebra_dataset_path,dataset_name,obj_name,f'init_poses/{j}.txt'))
        delta_matrix[:3,3]*=0.001
        delta_matrix[3,3]=1


        src_pcd=o3d.geometry.PointCloud()
        extrinsic=delta_matrix#delta_matrix@matrix
        R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
        cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
        raster_settings = RasterizationSettings(image_size=(int(image_size[0][0]),int(image_size[0][1])), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(obj_mesh.to(device))
        depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
        depth[depth==-1]=0
        depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)
        src_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_render_{}.ply'.format(source_dir,j),src_pcd)

        np.savetxt('{}/{}.txt'.format(gt_pose_dir,j),matrix)
        np.savetxt('{}/{}.txt'.format(init_pose_dir,j),extrinsic)

def evenly_distributed_rotation(n, random_seed=None):
    """
    uniformly sample N examples on a sphere
    """
    def normalize(vector, dim: int = -1):
        return vector / torch.norm(vector, p=2.0, dim=dim, keepdim=True)

    if random_seed is not None:
        torch.manual_seed(random_seed) # fix the sampling of viewpoints for reproducing evaluation

    indices = torch.arange(0, n, dtype=torch.float32) + 0.5

    phi = torch.acos(1 - 2 * indices / n)
    theta = math.pi * (1 + 5 ** 0.5) * indices
    points = torch.stack([
        torch.cos(theta) * torch.sin(phi), 
        torch.sin(theta) * torch.sin(phi), 
        torch.cos(phi),], dim=1)
    forward = -points
    
    down = normalize(torch.randn(n, 3), dim=1)
    right = normalize(torch.cross(down, forward))
    down = normalize(torch.cross(forward, right))
    R_mat = torch.stack([right, down, forward], dim=1)
    return R_mat

def depth_to_pointcloud(depth, K):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack([xs, ys, zs], axis=1)
    return pts


from ply2obj import ply2obj
import glob

def generate_standard_dataset(root_dir):         
    model_path_list=sorted(glob.glob(os.path.join(root_dir,'models/*/*.ply')))
    for model_path in model_path_list:
        if not os.path.exists(model_path.replace('ply','obj')):
            ply2obj(model_path,model_path.replace('ply','obj'))
        rot_intervals=[[30,90]]
        trans_intervals=[[0.1,0.3]]
        print('Generating model')
        for rot_interval,trans_interval in zip(rot_intervals,trans_intervals):
            generate_standard_render(root_dir,model_path.replace('ply','obj'),150,rot_interval,trans_interval)

def generate_noise_dataset(root_dir):
    model_path_list=glob.glob(os.path.join(root_dir,'models/*/*.ply'))
    for model_path in model_path_list:
        if not ('lm' in model_path or 'tless' in model_path or 'ycbv' in model_path): continue
        if not os.path.exists(model_path.replace('ply','obj')):
            ply2obj(model_path,model_path.replace('ply','obj'))
        generate_noise_render(root_dir,model_path.replace('ply','obj'),200,[10,70],[0.05,0.2])

def generate_scale_dataset(root_dir):
    model_path_list=sorted(glob.glob(os.path.join(root_dir,'models/*/*.ply')))
    for model_path in model_path_list:
        if not ('lm' in model_path or 'tless' in model_path or 'ycbv' in model_path): continue
        if not os.path.exists(model_path.replace('ply','obj')):
            ply2obj(model_path,model_path.replace('ply','obj'))
        scale_intervals=[[0.5,1.5]]
        for scale_interval in scale_intervals:
            print('Generating model')
            generate_scale_render(root_dir,model_path.replace('ply','obj'),200,[10,70],[0.05,0.2],scale_interval)

def generate_real_dataset(root_dir):         
    model_path_list=sorted(glob.glob(os.path.join(root_dir,'models/*/*.ply')))
    for model_path in model_path_list:
        if not os.path.exists(model_path.replace('ply','obj')):
            ply2obj(model_path,model_path.replace('ply','obj'))
        print('Generating model')
        generate_real_render(root_dir,model_path.replace('ply','obj'),150)
        # break


if __name__=='__main__':
    root_dir='./simreal_datasets'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str,default='./simreal_datasets',help='Root dir of simreal_datasets')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'noise','scale','real'], help='Evaluation mode')
    args=parser.parse_args()
    mode=args.mode
    # import pdb;pdb.set_trace()
    globals()[f'generate_{mode}_dataset'](root_dir)

