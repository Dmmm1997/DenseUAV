<h1 align="center"> Vision-Based UAV Self-Positioning in Low-Altitude Urban Environments </h1>

This repository contains code and dataset for the paper titled [Vision-Based UAV Self-Positioning in Low-Altitude Urban Environments](https://arxiv.org/abs/2201.09201). In this paper, we propose a method for accurately self-positioning unmanned aerial vehicles (UAVs) in challenging low-altitude urban environments using vision-based techniques. We provide the DenseUAV dataset and a Baseline model implementation to facilitate research in this task. Thank you for your kind attention.

![](https://github.com/Dmmm1997/DenseUAV/blob/main/docs/images/data.jpg)

![](https://github.com/Dmmm1997/DenseUAV/blob/main/docs/images/framework.jpg)

![](https://github.com/Dmmm1997/DenseUAV/blob/main/docs/images/model.png)

## News

- **`2025/9/22`**: Released a new **memory-efficient paradigm**, [SWA-PF](https://github.com/YuanJiayuuu/SWA-PF), the video demo is [Here](https://www.bilibili.com/video/BV1bzQJYzEnA?buvid=Y041B702FEB2BC484E06BA719B103480B87F&from_spmid=main.space-contribution.0.0&is_story_h5=false&mid=RqAphnnHLE9s3Yk5BetVLA%3D%3D&plat_id=116&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=DA295D81-6597-4EAF-B160-45A8BB2DC64D&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1762745583&unique_k=SL0nVLR&up_id=315497897&vd_source=5286ac14bb3e955efb5f90832eb06686).
- **`2024/8/28`**: Code and model released for a novel UAV self-localization paradigm named [DRL](https://github.com/Dmmm1997/DRL).
- **`2023/12/18`**: Our paper is accepted by IEEE Trans on Image Process.
- **`2023/8/14`**: Our dataset and code are released.

## Table of contents

- [News](#news)
- [Table of contents](#table-of-contents)
- [About Dataset](#about-dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset \& Preparation](#dataset--preparation)
- [Train \& Evaluation](#train--evaluation)
  - [Training and Testing](#training-and-testing)
  - [Evaluation](#evaluation)
- [Supported Methods](#supported-methods)
- [License](#license)
- [Citation](#citation)
- [Related Work](#related-work)

## About Dataset

The dataset split is as follows:
| Subset | UAV-view | Satellite-view | Classes | universities |
| -------- | ----- | ---- | ---- | ---- |
| Training | 6,768 | 13,536 | 2,256 | 10 |
| Query | 2,331 | 4,662 | 777 | 4 |
| Gallery | 9099 | 18198 | 3033 | 14 |

More detailed file structure:

```
├── DenseUAV/
│   ├── Dense_GPS_ALL.txt           /* format as: path latitude longitude height
│   ├── Dense_GPS_test.txt
│   ├── Dense_GPS_train.txt
│   ├── train/
│       ├── drone/                   /* drone-view training images
│           ├── 000001
│               ├── H100.JPG
│               ├── H90.JPG
│               ├── H80.JPG
|           ...
│       ├── satellite/               /* satellite-view training images
│           ├── 000001
│               ├── H100_old.tif
│               ├── H90_old.tif
│               ├── H80_old.tif
│               ├── H100.tif
│               ├── H90.tif
│               ├── H80.tif
|           ...
│   ├── test/
│       ├── query_drone/             /* UAV-view testing images
│       ├── query_satellite/         /* satellite-view testing images
```

## Prerequisites

- Python 3.7+
- GPU Memory >= 8G
- Numpy 1.21.2
- Pytorch 1.10.0+cu113
- Torchvision 0.11.1+cu113

## Installation

It is best to use cuda version 11.3 and pytorch version 1.10.0. You can download the corresponding version from this [website](https://download.pytorch.org/whl/torch_stable.html) and install it through `pip install`. Then you can execute the following command to install all dependencies.

```
pip install -r requirments.txt
```

Create the directory for saving the training log and ckpts.

```
mkdir checkpoints
```

## Dataset & Preparation

Download DenseUAV [HF_DATA](https://huggingface.co/datasets/Dmmm997/DenseUAV).

## Train & Evaluation

### Training and Testing

You could execute the following command to implement the entire process of training and testing.

```
bash train_test_local.sh
```

The setting of parameters in **train_test_local.sh** can refer to [Get Started](https://github.com/Dmmm1997/DenseUAV/blob/main/docs/training_parameters.md).

### Evaluation

The following commands are required to evaluate Recall and SDM separately.

```
cd checkpoints/<name>
python test.py --name <name> --test_dir <dir/to/testdir/of/dataset> --gpu_ids 0 --num_worker 4
```

the `<name>` is the dir name in your training setting, you can find in the `checkpoints/`.

**For Recall**

```
python evaluate_gpu.py
```

**For SDM**

```
python evaluateDistance.py --root_dir <dir/to/root/of/dataset>
```

We also provide the baseline checkpoints, [quark](https://pan.quark.cn/s/3ced42633793) [one-drive](https://seunic-my.sharepoint.cn/:u:/g/personal/230238525_seu_edu_cn/EUFoYjIdK_JNuxmvpb5QjLcB1hUHyedGwOnT3wTeN7Zqdg?e=LZuUxz).

```
unzip <file.zip> -d checkpoints
cd checkpoints/baseline
python test.py --test_dir <dataset_root>/test
python evaluate_gpu.py
python evaluateDistance.py --root_dir <dataset_root>
```

## Supported Methods

| <u>Augment</u>    | <u>Backbone</u> | <u>Head</u> | <u>Loss</u>                |
| ----------------- | --------------- | ----------- | -------------------------- |
| Random Rotate     | ResNet          | MaxPool     | CrossEntropy Loss.         |
| Random Affine     | EfficientNet    | AvgPool     | Focal Loss                 |
| Random Brightness | ConvNext        | MaxAvgPool  | Triplet Loss               |
| Random Erasing    | DeiT            | GlobalPool  | Hard-Mining Triplet Loss   |
|                   | PvT             | GemPool     | Same-Domain Triplet Loss   |
|                   | SwinTransformer | LPN         | Soft-Weighted Triplet Loss |
|                   | ViT             | FSRA        | KL Loss                    |

## License

This project is licensed under the [Apache 2.0 license](https://github.com/Dmmm1997/DenseUAV//blob/main/LICENSE).

## Citation

```bibtex
@ARTICLE{DenseUAV,
  author={Dai, Ming and Zheng, Enhui and Feng, Zhenhua and Qi, Lei and Zhuang, Jiedong and Yang, Wankou},
  journal={IEEE Transactions on Image Processing},
  title={Vision-Based UAV Self-Positioning in Low-Altitude Urban Environments},
  year={2024},
  volume={33},
  number={},
  pages={493-508},
  doi={10.1109/TIP.2023.3346279}}
```

## Related Work

- University-1652 [https://github.com/layumi/University1652-Baseline](https://github.com/layumi/University1652-Baseline)
- FSRA [https://github.com/Dmmm1997/FSRA](https://github.com/Dmmm1997/FSRA)
