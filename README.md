# Subway Health Classification

A classifier can be used for subway door health diagnosis.
This project is very lightweight, and uses codes of [mmcls1.x](https://github.com/open-mmlab/mmclassification/tree/1.x) & [timm](https://github.com/rwightman/pytorch-image-models).

## Install

1. Pytorch
   ```shell
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
   ```
   it's also recommended to build it in docker `pytorch/pytorch   1.11.0-cuda11.3-cudnn8-devel`
2. CUDA 11.3
   If you use docker above, you can skip this. Check CUDA version through `nvcc -V`
   ```shell
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2021 NVIDIA Corporation
    Built on Mon_May__3_19:15:13_PDT_2021
    Cuda compilation tools, release 11.3, V11.3.109
    Build cuda_11.3.r11.3/compiler.29920130_0
   ```
4. MMEngine & MMCV
   ```shell
   pip install openmim
   mim install mmengine mmcv
   ```
5. Requirements and the project itself
   ```shell
   pip install -v -e .
   ```

## Usage

### Prepare Data

The raw data is `dms1118c_clean.csv`, you can create a soft link at `SubwayHealthCls/data/subway/` which contains the raw data,
and then process it with the following scipts
```python
from mmcls.dataset import SubwayDataset
csv_file = 'path_to_raw_csv_file'   # recommend use abs path
out_dir = 'paht_to_output_dir'
SubwayDataset.split_data(csv_file, out_dir, train_ratio=0.1, shuffle=True)
```
**Please REMEMBER** to set your `data_path` in config `./configs/subway_transformer/subwayformer.py`

### Train

Please check scripts in `./my_tools/easy_train.py`, just run it directly or use below command
```python
python my_tools/easy_train.py
```

### Val

Please check scripts in `./my_tools/easy_val.py`, after setting the `ckpt` path, just run the python file or use below command
```python
python my_tools/easy_val.py
```