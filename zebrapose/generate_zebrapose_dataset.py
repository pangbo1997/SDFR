import glob
import os

dataset='lmo'
dataset_path=f'/home/pb/Bop_dataset/{dataset}/models'
lmo_obj = [
    'ape',
    'benchvise',
    'bowl', 
    'cam',
    'can',
    'cat',
    'cup',
    'driller',
    'duck',
    'eggbox',
    'glue',
    'holepuncher',
    'iron',
    'lamp',
    'phone'
]


for obj_file in glob.glob(f'{dataset_path}/*.ply'):
    obj_no=int(obj_file.split('/')[-1].split('.')[0].split('_')[-1])-1
    obj_name=lmo_obj[obj_no]
    ckpt_file=glob.glob(f'checkpoints/{dataset}_effnet/{obj_name}/*')[0]
    os.system(f'python test_cus.py --cfg config/config_BOP_effnet/{dataset}/{dataset}_BOP_effnet_test.txt --obj_name {obj_name} --ckpt_file {ckpt_file} --ignore_bit 0 --eval_output_path ./report')

dataset='ycbv'
dataset_path=f'/home/pb/Bop_dataset/{dataset}/models'
ycbv_obj = [
    'master_chef_can',
    'cracker_box',
    'sugar_box',
    'tomato_soup_can',
    'mustard_bottle',
    'tuna_fish_can',
    'pudding_box',
    'gelatin_box',
    'potted_meat_can',
    'banana',
    'pitcher_base',
    'bleach_cleanser',
    'bowl',
    'mug',
    'power_drill',
    'wood_block',
    'scissors',
    'large_marker',
    'large_clamp',
    'extra_large_clamp',
    'foam_brick',
]
for obj_file in glob.glob(f'{dataset_path}/*.ply'):
    obj_no=int(obj_file.split('/')[-1].split('.')[0].split('_')[-1])-1
    obj_name=ycbv_obj[obj_no]
    ckpt_file=glob.glob(f'checkpoints/{dataset}_effnet/{obj_name}/best_score/*')[0]
    os.system(f'python test_cus.py --cfg config/config_BOP_effnet/{dataset}/{dataset}_BOP_effnet_test.txt --obj_name {obj_name} --ckpt_file {ckpt_file} --ignore_bit 0 --eval_output_path ./report')


dataset='tless'
dataset_path=f'/home/pb/Bop_dataset/{dataset}/models_cad'
tless_obj = [
    'obj01',
    'obj02',
    'obj03',
    'obj04',
    'obj05',
    'obj06',
    'obj07',
    'obj08',
    'obj09',
    'obj10',
    'obj11',
    'obj12',
    'obj13',
    'obj14',
    'obj15',
    'obj16',
    'obj17',
    'obj18',
    'obj19',
    'obj20',
    'obj21',
    'obj22',
    'obj23',
    'obj24',
    'obj25',
    'obj26',
    'obj27',
    'obj28',
    'obj29',
    'obj30'
]


for obj_file in glob.glob(f'{dataset_path}/*.ply'):
    obj_no=int(obj_file.split('/')[-1].split('.')[0].split('_')[-1])-1
    obj_name=tless_obj[obj_no]
    ckpt_file=glob.glob(f'checkpoints/{dataset}_effnet/{dataset}/{obj_name}/best_score/*')[0]
    os.system(f'python test_cus.py --cfg config/config_BOP_effnet/{dataset}/{dataset}_BOP_effnet_test.txt --obj_name {obj_name} --ckpt_file {ckpt_file} --ignore_bit 0 --eval_output_path ./report')
