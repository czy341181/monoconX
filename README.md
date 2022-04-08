## Introduction

This repository is an unofficial implementation of the paper MonoCon for personal study.
We will continuously update it for better performance.

This repo benefits from [MonoDLE](https://github.com/xinzhuma/monodle) and [MonoCon](https://github.com/Xianpeng919/MonoCon).

## Usage

### Installation
This repo is tested on our local environment (python=3.7, cuda=10.1, pytorch=1.5.1), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n monodle python=3.7
```
Then, activate the environment:
```bash
conda activate mono3d
```

Install  Install PyTorch:

```bash
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
```

and other  requirements:
```bash
pip install -r requirements.txt
```

### Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT
  |data/
    |KITTI/
      |ImageSets/ [already provided in this repo]
      |object/			
        |training/
          |calib/
          |image_2/
          |label/
        |testing/
          |calib/
          |image_2/
```

### Training & Evaluation

Move to the workplace and train the network:

```sh
 cd #ROOT
 cd experiments/example
 python ../../tools/train_val.py --config kitti_example.yaml
```
The model will be evaluated automatically if the training completed. If you only want evaluate your trained model (or the provided [pretrained model](https://drive.google.com/file/d/1jaGdvu_XFn5woX0eJ5I2R6wIcBLVMJV6/view?usp=sharing)) , you can modify the test part configuration in the .yaml file and use the following command:

```sh
python ../../tools/train_val.py --config kitti_example.yaml --e
```

