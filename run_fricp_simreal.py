import open3d as o3d
import os
from scipy.spatial.transform import Rotation 
import numpy as np
import json
from tqdm import tqdm
import torch
import glob

from pytorch3d.io import load_objs_as_meshes, load_obj
import misc, visibility
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

from eval_metric import cal_vsd_error
def run_FRICP(method_no,data_loader,dataset_dir,result_file,model_path, render=False):
    record={}
    R_error_list=[]
    t_error_list=[]
    pred_pose_list=[]
    ADD_list=[]
    ADDS_list=[]
    VSD_list=[]

    mesh_model=load_objs_as_meshes([model_path], device=torch.device('cuda'))
    mesh_model.scale_verts_(0.001)
    diameter_json=json.load(open('/'.join(model_path.split('/')[:-1])+'/models_info.json'))
    obj_id=int(model_path.split('/')[-1][:-4].split('_')[-1])
    diameter=diameter_json[str(obj_id)]['diameter']/1000

    for src_pts_file,trg_pts_file,src_pts,trg_pts,init_pose,gt_pose in tqdm(data_loader):
        src_pts_file=src_pts_file[0]
        trg_pts_file=trg_pts_file[0]
        init_pose=init_pose[0].numpy()
        gt_pose=gt_pose[0].numpy()
        result_dir='/'.join(result_file.split('/')[:-1])
        tmp_txt_name=result_dir+result_file.split('/')[-1][:-5]
        if render: tmp_txt_name+='render'

        trg_pcd=o3d.io.read_point_cloud(trg_pts_file)
        src_pcd=o3d.io.read_point_cloud(src_pts_file)

        write_ply(f'{tmp_txt_name}trg_tmp.ply',trg_pcd)
        write_ply(f'{tmp_txt_name}src_tmp.ply',src_pcd)
        os.system(f'./FRICP {tmp_txt_name}trg_tmp.ply {tmp_txt_name}src_tmp.ply {tmp_txt_name} {method_no} > /dev/null 2>&1')
        os.system(f'rm -rf {tmp_txt_name}trg_tmp.ply {tmp_txt_name}src_tmp.ply')


        eps=1e-6

        pred_pose=np.loadtxt(f'{tmp_txt_name}m{method_no}trans.txt')@init_pose


        pred_pose_list.append(pred_pose.astype(np.float32).tolist())
        r_loss=(np.trace(pred_pose[:3,:3]@gt_pose[:3,:3].T)-1)/2
        r_loss=np.clip(r_loss,a_min=-1+eps,a_max=1-eps)
        R_error=(np.arccos(r_loss)*180/np.pi).astype(np.float32).tolist()
        t_error=(np.linalg.norm(pred_pose[:3,3]-gt_pose[:3,3])).astype(np.float32).tolist()

        R_error_list.append(R_error)
        t_error_list.append(t_error)
        os.system(f'rm -rf {tmp_txt_name}m{method_no}trans.txt {tmp_txt_name}m{method_no}reg_pc.ply')

        ADD_list.append(cal_add_error(pred_pose, gt_pose, mesh_model.verts_list()[0].detach().cpu().numpy()))

        if 'tless' in model_path:
                VSD_list.append(cal_vsd_error(torch.tensor(pred_pose), torch.tensor(gt_pose), mesh_model))

    record['pred_pose_list']=pred_pose_list
    record['5deg,5cm']=(((np.array(R_error_list)<5) & (np.array(t_error_list)<0.05)).sum()).tolist()
    record['ADD']=(np.array(ADD_list)<0.1*diameter).sum()/len(record['pred_pose_list']) 
    if 'tless' in model_path:
            record['VSD']=(np.array(VSD_list)<0.3).sum()/len(record['pred_pose_list']) 

    with open(result_file,'w') as f:
        json.dump(record,f,indent=2)
        
    # import pdb;pdb.set_trace()





from torch.utils.data import Dataset,DataLoader
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



class Visual_Dataset(Dataset):
    def __init__(self,data_dir,render=False):
        self.data_dir=data_dir

        self.source_file_dir=os.path.join(data_dir,'source')
        self.target_file_dir=os.path.join(data_dir,'target')
        self.target_file_list=glob.glob(os.path.join(data_dir,'target/*.ply'))

        gt_file_list_path=os.path.join(data_dir,'gt_file_list.json')
        if os.path.exists(gt_file_list_path):
            self.gt_file_list=json.load(open(gt_file_list_path))
        else:
            self.gt_file_list=[i.replace('target','gt_pose').replace('model_','').replace('.ply','.txt') for i in self.target_file_list]
            with open(gt_file_list_path,'w') as f: json.dump(self.gt_file_list,f)
        
        self.render=render
        
    def __getitem__(self, index):


        model_no=self.gt_file_list[index].split('/')[-1].split('_')[-1].split('.')[0]
        target_file=os.path.join(self.target_file_dir,f'model_{model_no}.ply')
        if self.render:
            source_file=os.path.join(self.source_file_dir,f'model_render_{model_no}.ply')
        else:
            source_file=os.path.join(self.source_file_dir,f'model_{model_no}.ply')

        trg_pts=np.array(o3d.io.read_point_cloud(target_file).points)
        src_pts=np.array(o3d.io.read_point_cloud(source_file).points)

        trg_pts=trg_pts[np.random.choice(np.arange(trg_pts.shape[0]),5000,replace=True)]
        src_pts=src_pts[np.random.choice(np.arange(src_pts.shape[0]),5000,replace=True)]

        gt_pose=np.loadtxt(os.path.join(self.data_dir,f'gt_pose/{model_no}.txt'))
        init_pose=np.loadtxt(os.path.join(self.data_dir,f'init_pose/{model_no}.txt'))
        
        return source_file,target_file,src_pts.astype(np.float32),trg_pts.astype(np.float32),init_pose.astype(np.float32),gt_pose.astype(np.float32)

    def __len__(self):
        return len(self.gt_file_list)

def eval_FRICP(args):

    root_dir=args.root_dir
    results_path=args.results_path
    mode=args.mode
    method_no=args.method_no
    render=args.render

    model_path_list=glob.glob(os.path.join(root_dir,'models/*/*.obj'))
    
    for model_path in model_path_list:
        render_path=os.path.join(root_dir,f'render/{mode}','_'.join(model_path.split('/')[-2:])[:-4])

        if not os.path.exists(render_path): continue

        result_path='{}/fricp_{}/{}/{}'.format(results_path,method_no,mode,'_'.join(model_path.split('/')[-2:])[:-4])
        os.makedirs(result_path,exist_ok=True)
        for _dir in os.listdir(render_path):
            
            dataset_dir=os.path.join(render_path,_dir)
            if render:
                result_file=os.path.join('{}/{}_render.json'.format(result_path,_dir))
            else:
                result_file=os.path.join('{}/{}_norender.json'.format(result_path,_dir))

            # if not os.path.exists(result_file): 
            dataset=Visual_Dataset(dataset_dir,render=render)
            dataloader=DataLoader(dataset,batch_size=1,shuffle=False,drop_last=False)
            run_FRICP(method_no,dataloader,dataset_dir,result_file,model_path,render=render)



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str,default='./simreal_datasets',help='Root dir of datasets')
    parser.add_argument('--results-path', type=str,default='./simreal_results',help='Results dir')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'noise','scale','real'], help='Evaluation mode')
    parser.add_argument('--method-no', type=int, default=3)
    parser.add_argument('--render', action='store_true', help='Use render')
    args=parser.parse_args()

    eval_FRICP(args)

