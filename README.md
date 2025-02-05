# Fast and Accurate 6D Object Pose Refinement via Implicit Surface Optimization

Code for **Fast and Accurate 6D Object Pose Refinement via
Implicit Surface Optimization**. 
This repository will continue to be updated.

## Installation
```
conda create -n sdfr python=3.8
pip install -r requirement.txt
cd lib/extensions
chmod +x build_ext.sh && ./build_ext.sh
```


# Sync Dataset Preparation & Running Command

For quick start, you can download a small dataset [here](https://drive.google.com/file/d/1YDvhBv6z5SByF_WaTQVzzL9qz3TyEm6a/view?usp=drive_link) and unzip it in the current project directory.

To prepar the full dataset, you need to first download the mesh models from the BOP challenge(Specifically, we will use the models from models_eval). And put them with the following structure:
```bash
.
├── datasets
│   ├── models
│   │   ├── lm
│   │       └── models_info.json
│   │       └── obj_000001.ply
│   ├── render
│   │   ...
│   
```
Then run the following script to generate the datasets:
```
python prepare_dataset.py --root-dir [datasets path] --mode ['standard' 'noise','scale','diverse','corrupt'] 
```
The results will be saved in the render directory.

## Run the pipline of SDFR

```
python run_sdfr.py --root-dir [datasets path] --mode ['standard' 'noise','scale','diverse','corrupt']
```
## Run the pipeline of FRICP
Before starting, you need to access their official[repository ](https://github.com/yaoyx689/Fast-Robust-ICP)  to compile the executable file.
```
python run_fricp.py --root-dir [datasets path] --mode ['standard' 'noise','scale','diverse','corrupt'] --method-no 3 --render
```
The definition of '--method-no' is the same as defined in FRICP, and '--render' refers to using the render operation as mentioned in the paper 

## Run the pipeline of Symmetric-ICP
Access the official [repository](https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2019_ASO/index.php) and use our "icpconverenge.cc" to compile the executable file.
```
python run_symmicp.py --root-dir [datasets path] --mode ['standard' 'noise','scale','diverse','corrupt'] --render
```
## Evaluation
The above command will predict poses in JSON files. You can use the following script to calculate the metrics defined in the paper. Our results are available [here](https://drive.google.com/file/d/1luA5QuBI9s7wsvDbWqgFpvwUUSJSGSWn/view?usp=drive_link).
```
python eval_metric.py
```

# SimReal Dataset Preparation & Running Command

The primary difference between the Sync dataset and the SimReal dataset lies in the source of the camera point clouds. In the Sync dataset, the point clouds are derived from rendered depth images, whereas in the SimReal dataset, they are generated from real depth images combined with predicted masks. 

To prepare this dataset, you first need to download the official test set from the [Bop Challenge](https://bop.felk.cvut.cz/datasets/).  The masks are generated following the approach used in [ZebraPose](https://github.com/suyz526/ZebraPose). To integrate our method, place our code into the zebrapose directory within their repository and execute the following command to generate the initial ZebraPose dataset.
```
python generate_zebrapose_dataset.py  
```
Then, organize the mesh models in a similar manner as described in Sync Dataset.
```bash
.
├── simreal_datasets
│   ├── models
│   │   ├── lm
│   │       └── models_info.json
│   │       └── obj_000001.ply
│   ├── render
│   │   ...
│   
```
Finally, run the following script to generate the SimReal dataset:
```
python prepare_simreal_dataset.py --root-dir [datasets path] --mode ['standard' 'noise','scale'] 
```

## Run the pipline of SDFR/FRICP/SymmICP

```
python run_sdfr_simreal.py --root-dir [datasets path] --mode ['standard' 'noise','scale']

python run_fricp_simreal.py --root-dir [datasets path] --mode ['standard' 'noise','scale'] --method-no 3 --render

python run_symmicp_simreal.py --root-dir [datasets path] --mode ['standard' 'noise','scale'] --render
```
## Acknowledgements

We would like to thank the authors of the following repository for their great work

https://github.com/nv-tlabs/nglod

https://github.com/yaoyx689/Fast-Robust-ICP

https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2019_ASO/index.php

https://github.com/dingdingcai/OVE6D-pose

https://github.com/zju3dv/OnePose_Plus_Plus