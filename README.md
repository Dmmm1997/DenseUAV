<h1 align="center"> Vision-Based UAV Self-Positioning in Low-Altitude Urban Environments </h1>

This repository contains code and dataset for the paper titled [Vision-Based UAV Self-Positioning in Low-Altitude Urban Environments](https://arxiv.org/abs/2201.09201). In this paper, we propose a method for accurately self-positioning unmanned aerial vehicles (UAVs) in challenging low-altitude urban environments using vision-based techniques. We provide the DenseUAV dataset and a Baseline model implementation to facilitate research in this task. Thank you for your kind attention.

![](https://github.com/Dmmm1997/DenseUAV/blob/main/docs/images/data.jpg)

![](https://github.com/Dmmm1997/DenseUAV/blob/main/docs/images/framework.jpg)

![](https://github.com/Dmmm1997/DenseUAV/blob/main/docs/images/model.png)

## News

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

Download DenseUAV upon request. You may use the request [Template](https://github.com/Dmmm1997/DenseUAV//blob/main/docs/Request.md).

## Train & Evaluation

### Training and Testing

You could execute the following command to implement the entire process of training and testing.

```
bash train_test_local.sh
```

The setting of parameters in **train_test_local.sh** can refer to [Get Started](https://github.com/Dmmm1997/DenseUAV//blob/main/docs/Get_started).

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

We also provide the baseline checkpoints, [ckpt](https://pan.quark.cn/s/3ced42633793).

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

The following paper uses and reports the result of the baseline model. You may cite it in your paper.

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
