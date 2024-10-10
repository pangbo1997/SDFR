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

    os.makedirs(source_dir,exist_ok=True)
    os.makedirs(target_dir,exist_ok=True)
    os.makedirs(init_pose_dir,exist_ok=True)
    os.makedirs(gt_pose_dir,exist_ok=True)

    obj_mesh=load_objs_as_meshes([model_path], device=torch.device('cuda'))
    image_size=torch.tensor([480,640]).reshape(1,2)
    device=torch.device('cuda')
    query_camera=np.array([[1598.5735940340862/3,0,  954.56/3],[0, 1598.5735940340862/3, 716.6687979261368/3],[0,         0,         1]])
    obj_mesh.scale_verts_(0.001)

    sample_view_point=evenly_distributed_rotation(num).cpu().numpy()

    obj_mesh_o3d=o3d.io.read_triangle_mesh(model_path)
    pcd = obj_mesh_o3d.sample_points_uniformly(number_of_points=5000)
    pcd.points=o3d.utility.Vector3dVector(np.array(pcd.points)/1000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))

    pts_array=np.array(pcd.points)
    pts_array_homo=np.concatenate((pts_array,np.ones(pts_array.shape[0])[:,None]),axis=1)
    for j in tqdm(range(num)):
        matrix=np.zeros((4,4))
        matrix[:3,:3]=sample_view_point[j]
        matrix[:3,3]=np.array([0,0,0.5])
        matrix[3,3]=1
        extrinsic=matrix

        R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
        cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
        raster_settings = RasterizationSettings(image_size=(480,640), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(obj_mesh.to(device))
        depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
        depth[depth==-1]=0
        depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)
        trg_pcd=o3d.geometry.PointCloud()
        trg_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
        trg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_{}.ply'.format(target_dir,j),trg_pcd)


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
        raster_settings = RasterizationSettings(image_size=(480,640), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(obj_mesh.to(device))
        depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
        depth[depth==-1]=0
        depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)

        src_pcd=o3d.geometry.PointCloud()
        src_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_render_{}.ply'.format(source_dir,j),src_pcd)

        src_pcd=o3d.geometry.PointCloud()
        src_pcd.points=o3d.utility.Vector3dVector((extrinsic@pts_array_homo.T).T[:,:3])
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_{}.ply'.format(source_dir,j),src_pcd)

        np.savetxt('{}/{}.txt'.format(gt_pose_dir,j),matrix)
        np.savetxt('{}/{}.txt'.format(init_pose_dir,j),extrinsic)
    
def generate_noise_render(root_dir,model_path,num,rot_interval,trans_interval,outlier_ratio):

    render_path=os.path.join(root_dir,'render/noise','_'.join(model_path.split('/')[-2:])[:-4])
    os.makedirs(render_path,exist_ok=True)

    source_dir='{}/R_{}_{}_t_{}_{}_noise_{}/source'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1],outlier_ratio)
    target_dir='{}/R_{}_{}_t_{}_{}_noise_{}/target'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1],outlier_ratio)
    init_pose_dir='{}/R_{}_{}_t_{}_{}_noise_{}/init_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1],outlier_ratio)
    gt_pose_dir='{}/R_{}_{}_t_{}_{}_noise_{}/gt_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1],outlier_ratio)

    if os.path.exists('{}/model_{}.ply'.format(source_dir,num-1)): return

    os.makedirs(source_dir,exist_ok=True)
    os.makedirs(target_dir,exist_ok=True)
    os.makedirs(init_pose_dir,exist_ok=True)
    os.makedirs(gt_pose_dir,exist_ok=True)

    obj_mesh=load_objs_as_meshes([model_path], device=torch.device('cuda'))
    image_size=torch.tensor([480,640]).reshape(1,2)
    device=torch.device('cuda')
    query_camera=np.array([[1598.5735940340862/3,0,  954.56/3],[0, 1598.5735940340862/3, 716.6687979261368/3],[0,         0,         1]])
    obj_mesh.scale_verts_(0.001)

    sample_view_point=evenly_distributed_rotation(num).cpu().numpy()

    obj_mesh_o3d=o3d.io.read_triangle_mesh(model_path)
    pcd = obj_mesh_o3d.sample_points_uniformly(number_of_points=5000)
    pcd.points=o3d.utility.Vector3dVector(np.array(pcd.points)/1000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))

    pts_array=np.array(pcd.points)
    pts_array_homo=np.concatenate((pts_array,np.ones(pts_array.shape[0])[:,None]),axis=1)
    for j in tqdm(range(num)):
        matrix=np.zeros((4,4))
        matrix[:3,:3]=sample_view_point[j]
        matrix[:3,3]=np.array([0,0,0.5])
        matrix[3,3]=1
        extrinsic=matrix

        R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
        cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
        raster_settings = RasterizationSettings(image_size=(480,640), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(obj_mesh.to(device))
        depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
        depth[depth==-1]=0
        depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)
        trg_pcd=o3d.geometry.PointCloud()
        trg_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
        trg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        # import pdb;pdb.set_trace()
        pts_array=np.array(trg_pcd.points)
        v_min,v_max=pts_array.min(axis=0),pts_array.max(axis=0)
        kdt = KDTree(pts_array, leaf_size=30, metric='euclidean')
        distance,ind=kdt.query(pts_array, k=6, return_distance=True)
        distance=np.median(distance,axis=1)
        normals=np.array(trg_pcd.normals)
        gauss_noise=normals*((distance)[:,None].repeat(3,axis=1))
        outlier=np.random.rand(int(outlier_ratio*pts_array.shape[0]),3)
        outlier=outlier*(v_max-v_min)+v_min
        pts_array+=gauss_noise
        pts_array=np.concatenate((pts_array,outlier),axis=0)
        trg_pcd.points=o3d.utility.Vector3dVector(pts_array)
        trg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_{}.ply'.format(target_dir,j),trg_pcd)
        # plt.imshow(depth)
        # plt.show()
        # import pdb;pdb.set_trace()

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

        extrinsic=delta_matrix
        R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
        cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
        raster_settings = RasterizationSettings(image_size=(480,640), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(obj_mesh.to(device))
        depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
        depth[depth==-1]=0
        src_pcd=o3d.geometry.PointCloud()
        src_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_render_{}.ply'.format(source_dir,j),src_pcd)

        src_pcd=o3d.geometry.PointCloud()
        src_pcd.points=o3d.utility.Vector3dVector((extrinsic@pts_array_homo.T).T[:,:3])
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_{}.ply'.format(source_dir,j),src_pcd)

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
    os.makedirs(source_dir,exist_ok=True)
    os.makedirs(target_dir,exist_ok=True)
    os.makedirs(init_pose_dir,exist_ok=True)
    os.makedirs(gt_pose_dir,exist_ok=True)

    obj_mesh=load_objs_as_meshes([model_path], device=torch.device('cuda'))
    image_size=torch.tensor([480,640]).reshape(1,2)
    device=torch.device('cuda')
    query_camera=np.array([[1598.5735940340862/3,0,  954.56/3],[0, 1598.5735940340862/3, 716.6687979261368/3],[0,         0,         1]])
    obj_mesh.scale_verts_(0.001)

    sample_view_point=evenly_distributed_rotation(num).cpu().numpy()

    obj_mesh_o3d=o3d.io.read_triangle_mesh(model_path)
    pcd = obj_mesh_o3d.sample_points_uniformly(number_of_points=5000)
    pcd.points=o3d.utility.Vector3dVector(np.array(pcd.points)/1000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))

    pts_array=np.array(pcd.points)
    pts_array_homo=np.concatenate((pts_array,np.ones(pts_array.shape[0])[:,None]),axis=1)
    for j in tqdm(range(num)):
        matrix=np.zeros((4,4))
        matrix[:3,:3]=sample_view_point[j]
        matrix[:3,3]=np.array([0,0,0.5])
        matrix[3,3]=1
        extrinsic=matrix

        scale=(np.random.rand()*(scale_interval[1]-scale_interval[0])+scale_interval[0])
        # obj_trg_mesh=load_objs_as_meshes(['lm_models/obj_{:06d}.obj'.format(i)], device=torch.device('cuda'))
        # obj_trg_mesh.scale_verts_(0.001)
        # obj_trg_mesh.scale_verts_(scale)

        R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
        cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
        raster_settings = RasterizationSettings(image_size=(480,640), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(obj_mesh.to(device))
        depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
        depth[depth==-1]=0
        depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)
        # cv2.imwrite('generated_models/lm_render_scale/scale_{}/obj_{:06d}/render/{}.png'.format(scale_interval[0],i,j),depth_color)
        trg_pcd=o3d.geometry.PointCloud()
        pts_array=depth_to_pointcloud(depth,query_camera)
        pts_array=(pts_array-matrix[:3,3])*scale+matrix[:3,3]
        # import pdb;pdb.set_trace()
        trg_pcd.points=o3d.utility.Vector3dVector(pts_array)
        trg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_{}.ply'.format(target_dir,j),trg_pcd)        # plt.imshow(depth)
        # plt.show()
        # import pdb;pdb.set_trace()

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
        extrinsic=delta_matrix


        R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
        cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
        raster_settings = RasterizationSettings(image_size=(480,640), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(obj_mesh.to(device))
        depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
        depth[depth==-1]=0
        src_pcd=o3d.geometry.PointCloud()
        src_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_render_{}.ply'.format(source_dir,j),src_pcd)

        src_pcd=o3d.geometry.PointCloud()
        src_pts_array_homo=pts_array_homo.copy()
        src_pcd.points=o3d.utility.Vector3dVector((extrinsic@src_pts_array_homo.T).T[:,:3])
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_{}.ply'.format(source_dir,j),src_pcd)

        np.savetxt('{}/{}.txt'.format(gt_pose_dir,j),matrix)
        np.savetxt('{}/{}.txt'.format(init_pose_dir,j),extrinsic)

def generate_diverse_render(root_dir,model_path,num,rot_interval,trans_interval):

    render_path=os.path.join(root_dir,'render/diverse','_'.join(model_path.split('/')[-2:])[:-4])
    os.makedirs(render_path,exist_ok=True)

    source_dir='{}/R_{}_{}_t_{}_{}/source'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    target_dir='{}/R_{}_{}_t_{}_{}/target'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    init_pose_dir='{}/R_{}_{}_t_{}_{}/init_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    gt_pose_dir='{}/R_{}_{}_t_{}_{}/gt_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])

    if os.path.exists('{}/model_{}.ply'.format(source_dir,num-1)): return

    os.makedirs(source_dir,exist_ok=True)
    os.makedirs(target_dir,exist_ok=True)
    os.makedirs(init_pose_dir,exist_ok=True)
    os.makedirs(gt_pose_dir,exist_ok=True)

    obj_mesh=load_objs_as_meshes([model_path], device=torch.device('cuda'))
    image_size=torch.tensor([480,640]).reshape(1,2)
    device=torch.device('cuda')
    query_camera=np.array([[1598.5735940340862/3,0,  954.56/3],[0, 1598.5735940340862/3, 716.6687979261368/3],[0,         0,         1]])
    obj_mesh.scale_verts_(0.001)

    sample_view_point=evenly_distributed_rotation(num).cpu().numpy()

    obj_mesh_o3d=o3d.io.read_triangle_mesh(model_path)
    pcd = obj_mesh_o3d.sample_points_uniformly(number_of_points=5000)
    pcd.points=o3d.utility.Vector3dVector(np.array(pcd.points)/1000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))

    pts_array=np.array(pcd.points)
    pts_array_homo=np.concatenate((pts_array,np.ones(pts_array.shape[0])[:,None]),axis=1)
    for j in tqdm(range(num)):
        matrix=np.zeros((4,4))
        matrix[:3,:3]=sample_view_point[j]
        matrix[:3,3]=np.array([0,0,0.5])
        matrix[3,3]=1
        extrinsic=matrix

        R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
        cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
        raster_settings = RasterizationSettings(image_size=(480,640), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(obj_mesh.to(device))
        depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
        depth[depth==-1]=0
        depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)
        trg_pcd=o3d.geometry.PointCloud()
        trg_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
        trg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_{}.ply'.format(target_dir,j),trg_pcd)


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
        raster_settings = RasterizationSettings(image_size=(480,640), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(obj_mesh.to(device))
        depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
        depth[depth==-1]=0
        depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)

        src_pcd=o3d.geometry.PointCloud()
        src_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_render_{}.ply'.format(source_dir,j),src_pcd)

        src_pcd=o3d.geometry.PointCloud()
        src_pcd.points=o3d.utility.Vector3dVector((extrinsic@pts_array_homo.T).T[:,:3])
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_{}.ply'.format(source_dir,j),src_pcd)

        np.savetxt('{}/{}.txt'.format(gt_pose_dir,j),matrix)
        np.savetxt('{}/{}.txt'.format(init_pose_dir,j),extrinsic)

import torchvision.transforms as transforms
erase_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=1, scale=(0.1, 0.15), ratio=(0.3, 3.3), value=0),
    transforms.RandomErasing(p=0.8, scale=(0.1, 0.2), ratio=(0.3, 3.3), value=0),
    transforms.RandomErasing(p=0.5, scale=(0.1, 0.2), ratio=(0.3, 3.3), value=0)
])

def process_occlusion_removal(depth):
    # depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)
    # cv2.imwrite('tmp.png',depth_color)
    v,u=depth.nonzero()
    x0,y0,x1,y1=u.min(),v.min(),u.max(),v.max()
    depth[y0:y1,x0:x1]=erase_transform(depth[y0:y1,x0:x1]).cpu().numpy()
    # depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)
    # cv2.imwrite('tmp2.png',depth_color)
    return depth

import random
def process_reflection_removal(pcd):
    # pass
    pcd.orient_normals_towards_camera_location(camera_location=[0.,0.,0.])
    normals=np.array(pcd.normals)
    camera_direction=np.array([0,0,-1])
    angles= np.arccos(normals@camera_direction.T)*180/np.pi
    filtered_indices = []
    for i, angle in enumerate(angles):
        if angle > 70 and random.random() > 0.8:  
            filtered_indices.append(i)
    filtered_pcd = pcd.select_by_index(filtered_indices,invert=True)
    return filtered_pcd

def generate_corrupt_render(root_dir,model_path,num,rot_interval,trans_interval):

    render_path=os.path.join(root_dir,'render/corrupt','_'.join(model_path.split('/')[-2:])[:-4])
    os.makedirs(render_path,exist_ok=True)

    source_dir='{}/R_{}_{}_t_{}_{}/source'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    target_dir='{}/R_{}_{}_t_{}_{}/target'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    init_pose_dir='{}/R_{}_{}_t_{}_{}/init_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])
    gt_pose_dir='{}/R_{}_{}_t_{}_{}/gt_pose'.format(render_path,rot_interval[0],rot_interval[1],trans_interval[0],trans_interval[1])

    if os.path.exists('{}/model_{}.ply'.format(source_dir,num-1)): return

    os.makedirs(source_dir,exist_ok=True)
    os.makedirs(target_dir,exist_ok=True)
    os.makedirs(init_pose_dir,exist_ok=True)
    os.makedirs(gt_pose_dir,exist_ok=True)

    obj_mesh=load_objs_as_meshes([model_path], device=torch.device('cuda'))
    image_size=torch.tensor([480,640]).reshape(1,2)
    device=torch.device('cuda')
    query_camera=np.array([[1598.5735940340862/3,0,  954.56/3],[0, 1598.5735940340862/3, 716.6687979261368/3],[0,         0,         1]])
    obj_mesh.scale_verts_(0.001)

    sample_view_point=evenly_distributed_rotation(num).cpu().numpy()

    obj_mesh_o3d=o3d.io.read_triangle_mesh(model_path)
    pcd = obj_mesh_o3d.sample_points_uniformly(number_of_points=5000)
    pcd.points=o3d.utility.Vector3dVector(np.array(pcd.points)/1000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))

    pts_array=np.array(pcd.points)
    pts_array_homo=np.concatenate((pts_array,np.ones(pts_array.shape[0])[:,None]),axis=1)
    for j in tqdm(range(num)):
        matrix=np.zeros((4,4))
        matrix[:3,:3]=sample_view_point[j]
        matrix[:3,3]=np.array([0,0,0.5])
        matrix[3,3]=1
        extrinsic=matrix

        R,T,cam_K=torch.tensor(extrinsic[:3,:3],dtype=torch.float32).reshape(1,3,3),torch.tensor(extrinsic[:3,3],dtype=torch.float32).reshape(1,3),torch.tensor(query_camera,dtype=torch.float32).reshape(1,3,3)
        cameras=cameras_from_opencv_projection(R=R,tvec=T,camera_matrix=cam_K,image_size=image_size).to(device)
        raster_settings = RasterizationSettings(image_size=(480,640), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(obj_mesh.to(device))
        depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
        depth[depth==-1]=0
        depth=process_occlusion_removal(depth)
        depth_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)


        trg_pcd=o3d.geometry.PointCloud()
        trg_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
        trg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        trg_pcd=process_reflection_removal(trg_pcd)
        write_ply('{}/model_{}.ply'.format(target_dir,j),trg_pcd)

       
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
        raster_settings = RasterizationSettings(image_size=(480,640), blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(obj_mesh.to(device))
        depth=fragments.zbuf[0,:,:,0].detach().cpu().numpy()
        depth[depth==-1]=0
        

        src_pcd=o3d.geometry.PointCloud()
        src_pcd.points=o3d.utility.Vector3dVector(depth_to_pointcloud(depth,query_camera))
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_render_{}.ply'.format(source_dir,j),src_pcd)

        src_pcd=o3d.geometry.PointCloud()
        src_pcd.points=o3d.utility.Vector3dVector((extrinsic@pts_array_homo.T).T[:,:3])
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        write_ply('{}/model_{}.ply'.format(source_dir,j),src_pcd)

        np.savetxt('{}/{}.txt'.format(gt_pose_dir,j),matrix)
        np.savetxt('{}/{}.txt'.format(init_pose_dir,j),extrinsic)
    
import math

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
    model_path_list=glob.glob(os.path.join(root_dir,'models/*/*.ply'))
    for model_path in model_path_list:
        if not ('lm' in model_path or 'tless' in model_path or 'ycbv' in model_path): continue
        if not os.path.exists(model_path.replace('ply','obj')):
            ply2obj(model_path,model_path.replace('ply','obj'))
        rot_intervals=[[5,30],[30,50],[50,70],[70,90]]
        trans_intervals=[[0.01,0.05],[0.05,0.1],[0.1,0.2],[0.2,0.3]]
        print('Generating model')
        for rot_interval,trans_interval in zip(rot_intervals,trans_intervals):
            generate_standard_render(root_dir,model_path.replace('ply','obj'),150,rot_interval,trans_interval)

def generate_noise_dataset(root_dir):
    model_path_list=glob.glob(os.path.join(root_dir,'models/*/*.ply'))
    for model_path in model_path_list:
        if not ('lm' in model_path or 'tless' in model_path or 'ycbv' in model_path): continue
        if not os.path.exists(model_path.replace('ply','obj')):
            ply2obj(model_path,model_path.replace('ply','obj'))
        outlier_ratios=[0.5,1.0]
        for outlier_ratio in outlier_ratios:
            print('Generating model')
            generate_noise_render(root_dir,model_path.replace('ply','obj'),200,[10,70],[0.05,0.2],outlier_ratio)

def generate_scale_dataset(root_dir):
    model_path_list=sorted(glob.glob(os.path.join(root_dir,'models/*/*.ply')))
    for model_path in model_path_list:
        if not ('lm' in model_path or 'tless' in model_path or 'ycbv' in model_path): continue
        if not os.path.exists(model_path.replace('ply','obj')):
            ply2obj(model_path,model_path.replace('ply','obj'))
        scale_intervals=[[1.2,1.5],[0.5,0.8]]
        for scale_interval in scale_intervals:
            print('Generating model')
            generate_scale_render(root_dir,model_path.replace('ply','obj'),200,[10,70],[0.05,0.2],scale_interval)


def generate_diverse_dataset(root_dir):              
    model_path_list=glob.glob(os.path.join(root_dir,'models/*/*.ply'))
    # import pdb;pdb.set_trace()
    for model_path in model_path_list:
        if not os.path.exists(model_path.replace('ply','obj')):
            ply2obj(model_path,model_path.replace('ply','obj'))
        rot_intervals=[[5,70]]
        trans_intervals=[[0.01,0.2]]
        print('Generating model')
        for rot_interval,trans_interval in zip(rot_intervals,trans_intervals):
            generate_diverse_render(root_dir,model_path.replace('ply','obj'),100,rot_interval,trans_interval)


def generate_corrupt_dataset(root_dir):              
    model_path_list=glob.glob(os.path.join(root_dir,'models/*/*.ply'))
    # import pdb;pdb.set_trace()
    for model_path in model_path_list:
        if not os.path.exists(model_path.replace('ply','obj')):
            ply2obj(model_path,model_path.replace('ply','obj'))
        rot_intervals=[[5,70]]
        trans_intervals=[[0.01,0.2]]
        print('Generating model')
        for rot_interval,trans_interval in zip(rot_intervals,trans_intervals):
            generate_corrupt_render(root_dir,model_path.replace('ply','obj'),100,rot_interval,trans_interval)


if __name__=='__main__':
    root_dir='./datasets'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str,default='./datasets',help='Root dir of datasets')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'noise','scale','diverse','corrupt'], help='Evaluation mode')
    args=parser.parse_args()
    mode=args.mode
    globals()[f'generate_{mode}_dataset'](root_dir)

