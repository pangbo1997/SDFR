import os
import numpy as np
from glob import glob
from plyfile import PlyData
from tqdm import tqdm


def write_obj(verts, faces, obj_path):
    """
    Write .obj file
    """
    assert obj_path[-4:] == '.obj'
    with open(obj_path, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces+1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def ply2obj(ply_name, obj_name):
    plydata = PlyData.read(ply_name)
    pc = plydata['vertex'].data
    # print(pc)
    faces = plydata['face'].data
    # print(faces)
    #import pdb;pdb.set_trace()
    try:
        pc_array = np.array([[x, y, z] for x,y,z,_,_,_,_,_,_,_ in pc])
    except:
        pc_array = np.array([[x, y, z] for x,y,z in pc])
    face_array = np.array([face[0] for face in faces], dtype=np.int64)
    write_obj(pc_array, face_array, obj_name)


import argparse
import glob
if __name__ == "__main__":
    #filelist = glob(r'路径\*.ply')
    # print(filelist)
    filelist=['obj_000016.ply']
    # filelist=glob.glob('*.ply')
    for i in tqdm(range(len(filelist))):
        if os.path.exists(filelist[i].replace('.ply','.obj')):
            continue
        ply2obj(filelist[i], filelist[i].replace('.ply','.obj')) #目标文件与源文件保存在同一目录
