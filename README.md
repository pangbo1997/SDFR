# Fast and Accurate 6D Object Pose Refinement via Implicit Surface Optimization

Code for **Fast and Accurate 6D Object Pose Refinement via
Implicit Surface Optimization**. 
This repository will continue to be updated.


## Dataset Preparation 

For quick start, you can download a small dataset [here](https://drive.google.com/file/d/1YDvhBv6z5SByF_WaTQVzzL9qz3TyEm6a/view?usp=drive_link) and unzip it in the current project directory.
The full dataset will be uploaded soon. 
## Installation
```
conda create -n sdfr python=3.8
pip install -r requirement.txt
cd lib/extensions
chmod +x build_ext.sh && ./build_ext.sh
```
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

## Acknowledgements

We would like to thank the authors of the following repository for their great work

https://github.com/nv-tlabs/nglod

https://github.com/yaoyx689/Fast-Robust-ICP

https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2019_ASO/index.php

https://github.com/dingdingcai/OVE6D-pose

https://github.com/zju3dv/OnePose_Plus_Plus