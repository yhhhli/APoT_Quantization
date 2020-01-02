# APoT Quantization

This repo contains the code and data of the following paper accepeted by [ICLR 2020](https://openreview.net/group?id=ICLR.cc/2020/Conference)

> [Additive Power-of-Two Quantization: A Non-uniform Discretization For Neural Networks](https://openreview.net/pdf?id=BkgXT24tDS)

**training codes** will be open sourced soon.

<p align="center">
  <img src="https://i.imgur.com/0oxm19W.png">
</p>

```
@inproceedings{Li2020Additive,
title={Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks},
author={Yuhang Li and Xin Dong and Wei Wang},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BkgXT24tDS}
}
```

## Installation

### Prerequisites

Pytorch 1.1.0 with CUDA

### Dataset Preparation

* Please prepare the ImageNet validation and training dataset, we use [official example code](https://github.com/pytorch/examples/blob/master/imagenet/main.py) here to provide dataloader. 
* The CIFAR10 dataset can be download automatically (update soon). 

## ImageNet

`models.quant_layer.py` contains the configuration for quantization. In particular, you can specify them in 

Then, you can get the output like this:

```Bash
=> creating model 'resnet18'
=> loading the 3-bit quantized model from checkpoint
weight alpha: 3.535139, act alpha: 0.484280
weight alpha: 2.505285, act alpha: 0.800512
weight alpha: 2.760515, act alpha: 1.560048
weight alpha: 2.363593, act alpha: 1.061420
weight alpha: 2.069195, act alpha: 2.241576
weight alpha: 2.578392, act alpha: 1.266224
weight alpha: 2.352476, act alpha: 1.673914
weight alpha: 2.559555, act alpha: 1.752608
weight alpha: 2.545749, act alpha: 1.267403
weight alpha: 2.758606, act alpha: 2.022375
weight alpha: 2.645624, act alpha: 1.342584
weight alpha: 3.033730, act alpha: 1.749131
weight alpha: 2.853369, act alpha: 1.704186
weight alpha: 2.850048, act alpha: 1.174385
weight alpha: 2.443813, act alpha: 1.799547
weight alpha: 1.989842, act alpha: 1.165739
weight alpha: 2.054534, act alpha: 1.376302
weight alpha: 1.699125, act alpha: 2.244684
weight alpha: 2.085015, act alpha: 1.283475
49078it [00:05, 9342.39it/s]
 Test: [ 0/49]  Time 28.736 (28.736)    Loss 1.2406e+00 (1.2406e+00)    Acc@1  69.73 ( 69.73)   Acc@5  89.45 ( 89.45)
Test: [10/49]   Time  0.160 ( 3.198)    Loss 1.1772e+00 (1.2201e+00)    Acc@1  72.07 ( 70.46)   Acc@5  90.14 ( 89.43)
Test: [20/49]   Time  0.161 ( 1.751)    Loss 1.1765e+00 (1.2404e+00)    Acc@1  70.90 ( 69.94)   Acc@5  89.75 ( 89.07)
Test: [30/49]   Time  0.159 ( 1.246)    Loss 1.2093e+00 (1.2423e+00)    Acc@1  69.43 ( 69.79)   Acc@5  89.94 ( 88.96)
Test: [40/49]   Time  0.339 ( 1.077)    Loss 1.2208e+00 (1.2504e+00)    Acc@1  70.51 ( 69.60)   Acc@5  90.33 ( 88.90)
 * Acc@1 69.672 Acc@5 88.986
```

