import open3d as o3d
import os
from scipy.spatial.transform import Rotation 
import numpy as np
import json
from tqdm import tqdm
import torch
from lib import rendering
import glob
from sdfr import *


import math
import argparse
import glob
import logging as log
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)
from lib.trainer import Trainer
from lib.options import parse_options

def init_SDFR(args,obj_mesh_file):

    sdfr_model=obj_mesh_file.replace('.obj','.pth')
    if not os.path.exists(sdfr_model):
        args.net='OctreeSDF'
        args.num_lods=5
        args.dataset_path=obj_mesh_file
        args.epochs=40
        args.model_path='/'.join(sdfr_model.split('/')[:-1])
        args.exp_name=sdfr_model.split('/')[-1][:-4]
        log.info(f'Training on {args.dataset_path}')
        model = Trainer(args, '')
        model.train()

    obj_mesh, _,_ = rendering.load_object(obj_mesh_file, resize=False, recenter=False)
    obj_mesh.rescale(0.001)
    V = torch.tensor(obj_mesh.vertices * 1000,dtype=torch.float32)
    V_max, _ = torch.max(V, dim=0)
    V_min, _ = torch.min(V, dim=0)
    V_center = (V_max + V_min) / 2.
    V = V - V_center
    max_dist = torch.sqrt(torch.max(torch.sum(V ** 2, dim=-1)))
    obj_scale = 1. / max_dist
    scale_info=obj_mesh_file.replace('.obj','_scale.txt')
    np.savetxt(scale_info,np.array([obj_scale,*V_center]))

def run_SDFR(args,sdf_net,center_scale_info,data_loader,result_file,refine_scale=False,sample_farthest=True):

    mode=args.mode
    if mode=='noise': sample_farthest=False
    if mode=='scale': refine_scale=True

    obj_scale,center_x,center_y,center_z=center_scale_info
    V_center=torch.tensor([center_x,center_y,center_z],dtype=torch.float32).cuda()
    eps=1e-6
    record={}
    pred_pose_list=[]
    R_error_list=[]
    t_error_list=[]
    for src_pts_file,trg_pts_file,src_pts,trg_pts,init_pose,gt_pose in tqdm(data_loader):
        # import pdb;pdb.set_trace()
        if refine_scale:
            initial_scale=0.5*torch.norm(trg_pts.max(dim=1)[0]-trg_pts.min(dim=1)[0],dim=1)/torch.norm(src_pts.max(dim=1)[0]-src_pts.min(dim=1)[0],dim=1)
            initial_scale=initial_scale.cuda()
        else:
            initial_scale=None

        R,t,z=SDFR_adam(sdf_net,trg_pts,obj_scale,init_pose,gt_pose,refine_scale=refine_scale,initial_scale=initial_scale,sample_farthest=sample_farthest)
        r_loss=(torch.einsum('bii->b',torch.bmm(R,gt_pose[:,:3,:3].permute(0,2,1)))-1)/2
        r_loss=torch.clip(r_loss,min=-1+eps,max=1-eps)
        R_error=torch.arccos(r_loss)*180/np.pi
        t_error=(t-gt_pose[:,:3,3]).norm(dim=1)
        # print('R_error:{},t_error:{}'.format(R_error.mean(),t_error.mean()))
        R_error_list.extend(R_error.numpy().tolist())
        t_error_list.extend(t_error.numpy().tolist())

        pred_pose=np.zeros((R.shape[0],4,4))
        pred_pose[3,3]=1
        pred_pose[:,:3,:3]=R.detach().cpu().numpy()
        pred_pose[:,:3,3]=t.detach().cpu().numpy()
        if refine_scale:
            pred_pose[:,3,3]=z.detach().cpu().numpy().squeeze()
        pred_pose_list.extend(pred_pose.tolist())


    record['pred_pose_list']=pred_pose_list
    record['5deg,5cm']=(((np.array(R_error_list)<5) & (np.array(t_error_list)<0.05)).sum()).tolist()

    with open(result_file,'w') as f:
        json.dump(record,f,indent=2)




from torch.utils.data import Dataset,DataLoader

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

def eval_SDFR(args):
    root_dir=args.root_dir
    mode=args.mode
    results_path=args.results_path
    model_path_list=glob.glob(os.path.join(root_dir,'models/*/*.obj'))

    if mode=='noise': batch_size=32
    else: batch_size=160

    for model_path in model_path_list:
        if mode=='standard' or mode=='noise' or mode=='scale':
            if 'lm' not in model_path and 'ycbv' not in model_path and 'tless' not in model_path:
                continue

        init_SDFR(args, model_path)
        sdf_net=get_sdf_net(args,model_path.replace('.obj','.pth'))
        center_scale_info=np.loadtxt(model_path.replace('.obj','_scale.txt'))

        render_path=os.path.join(root_dir,f'render/{mode}','_'.join(model_path.split('/')[-2:])[:-4])
        result_path='{}/sdfr/{}/{}'.format(results_path,mode,'_'.join(model_path.split('/')[-2:])[:-4])
        os.makedirs(result_path,exist_ok=True)
        for _dir in os.listdir(render_path):
            result_file=os.path.join('{}/{}.json'.format(result_path,_dir))
            if os.path.exists(result_file): continue
            dataset_dir=os.path.join(render_path,_dir)
            dataset=Visual_Dataset(dataset_dir)
            dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,drop_last=False)
            run_SDFR(args,sdf_net,center_scale_info,dataloader,result_file)


if __name__=='__main__':


    parser = parse_options(return_parser=True)
    parser.add_argument('--root-dir', type=str,default='./datasets',help='Root dir of datasets')
    parser.add_argument('--results-path', type=str,default='./results',help='Results dir')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'noise','scale','diverse','corrupt'], help='Evaluation mode')
    parser.add_argument('--gpu', type=int, default=0)
    args=parser.parse_args()
    # import pdb;pdb.set_trace()

    eval_SDFR(args)

