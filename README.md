# SOD_DEDDH

Small Object Detection Using Detail Enhancement and Decoupled Detection Head

# Preparation work

We have used the following versions of OS and softwares:

- OS:  Ubuntu 22.04
- Python: 3.8
- GPU: RTX3090Ti
- CUDA: 12.0
- PyTorch: 1.9.0+cu111
- TorchVision: 0.10.0+cu111
- TorchAudio: 0.9.0
- MMCV-FULL: 1.4.0
- MMDetection: 2.13.0

## Install

#### a. Create a conda virtual environment and activate it.

```shell
conda create -n sod python=3.8
conda activate sod
```

#### b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Install mmcv-full(we used mmcv-full==1.4.0)

```shell
pip install openmim
mim install mmcv-full==1.4.0
```

#### d. Install COCOAPI-AITOD for Evaluating on AI-TOD dataset(The VisDrone2019 dataset does not require it)

```shell
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"
```

You can also refer to [official instruction](https://github.com/jwwangchn/cocoapi-aitod) for installing COCOAPI-AITOD.

#### e. Install SOD-DEDDH

```shell
git clone https://github.com/SYLan2019/SOD-DEDDH.git
cd SOD-DEDDH
pip install -r requirements.txt
pip install -v -e .
# or "python setup.py install"
```

## Prepare datasets

We only provide the dataset in JSON format.

- VisDrone2019: 
- AITODv2: 

Our data folder structure is as follows:

```shell
SOD-DEDDH
├── mmdet
├── tools
├── configs
├── data
│   ├── AI-TOD
│   │   ├── annotations
│   │   │    │─── aitodv2_train.json
│   │   │    │─── aitodv2_test.json
|   |   |    |___ aitodv2_val.json
│   │   ├── train
|   |   |    |___images
|   |   |    |    |─── ***.png
|   |   |    |    |─── ***.png
│   │   ├── val
│   │   │    │─── imges
|   |   |    |    |─── ***.png
│   │   │    │    |─── ***.png
│   ├── VisDrone
│   │   ├── annotations
│   │   │    │─── VisDroneTrain.json
│   │   │    │─── VisDroneTest.json
|   |   |    |___ VisDroneVal.json
│   │   ├── train
|   |   |    |─── ***.png
|   |   |    |─── ***.png
│   │   ├── test
|   |   |    |─── ***.png
|   |   |    |─── ***.png
│   │   ├── val
|   |   |    |─── ***.png
|   |   |    |─── ***.png

```

If your data folder structure is different, you may need to change the corresponding paths in config files (configs/\_base\_/datasets/aitodv2_detection(visdrone).py).

## Run

The SOD-DEDDH's config files are in [configs/rfla/SOD_DEDDH](https://github.com/jwwangchn/NWD/tree/main/configs/SOD_DEDDH).

Please see MMDetection full tutorials [with existing dataset](docs/1_exist_data_model.md) for beginners.

#### Training(on a single GPU)

For the VisDrone2019 dataset.

```shell
python tools/train.py configs/rfla/
```

For the AITODv2 dataset.

```shell
python tools.train.py configs/rfla/
```

#### testing 

testing on VisDrone2019 validation set

```shell
python tools/test.py configs/rfla/ your_training_weight.pth --eval bbox
```

testing on AITODv2 test set

```shell
python tools/test.py configs/rfla/ your_training_weight.pth --eval bbox
```
