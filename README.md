# APoT Quantization

This repo contains the code and data of the following paper accepeted by [ICLR 2020](https://openreview.net/group?id=ICLR.cc/2020/Conference)

> [Additive Power-of-Two Quantization: A Non-uniform Discretization For Neural Networks](https://openreview.net/pdf?id=BkgXT24tDS)

**training codes** will be open sourced soon.

<p align="center">
  <img src="https://i.imgur.com/0oxm19W.png">
</p>

## Installation

### Prerequisites

Pytorch 1.1.0 with CUDA

### Dataset Preparation

* Please prepare the ImageNet validation dataset, we use [official example code](https://github.com/pytorch/examples/blob/master/imagenet/main.py) here to provide validation dataloader. 
* The CIFAR10 validation dataset can be download automatically. 

## CIFAR10

change to the CIFAR10 directory first, and then  

```bash
python test.py --bit 5 
```

Then, you will get the output like this:

```bash
=> Building model...
=> loading the 5-bit quantized model from checkpoint
weight alpha: 2.828116, act alpha: 3.621083
weight alpha: 2.414355, act alpha: 4.517607
weight alpha: 2.685967, act alpha: 5.627809
weight alpha: 2.298938, act alpha: 5.268279
weight alpha: 2.696353, act alpha: 5.807320
weight alpha: 2.834752, act alpha: 5.221708
weight alpha: 2.683185, act alpha: 5.928465
weight alpha: 2.768446, act alpha: 5.032526
weight alpha: 2.308132, act alpha: 6.052375
weight alpha: 2.562996, act alpha: 5.866919
weight alpha: 2.166837, act alpha: 5.420985
weight alpha: 2.441639, act alpha: 6.189524
weight alpha: 2.481792, act alpha: 5.468916
weight alpha: 2.854374, act alpha: 6.030290
weight alpha: 2.444138, act alpha: 4.387599
weight alpha: 2.275344, act alpha: 5.872911
weight alpha: 2.159221, act alpha: 5.372866
weight alpha: 2.109906, act alpha: 4.771915
weight alpha: 2.147920, act alpha: 5.499549
weight alpha: 2.642403, act alpha: 5.065750
=> loading cifar10 data...
Files already downloaded and verified
Test: [0/100]	Time 0.366 (0.366)	Loss 0.2783 (0.2783)	Prec 93.000% (93.000%)
Test: [10/100]	Time 0.017 (0.051)	Loss 0.3108 (0.3172)	Prec 91.000% (92.545%)
Test: [20/100]	Time 0.016 (0.036)	Loss 0.5455 (0.3373)	Prec 87.000% (92.000%)
Test: [30/100]	Time 0.016 (0.030)	Loss 0.3876 (0.3574)	Prec 91.000% (92.065%)
Test: [40/100]	Time 0.015 (0.026)	Loss 0.5160 (0.3585)	Prec 89.000% (91.976%)
Test: [50/100]	Time 0.015 (0.024)	Loss 0.2793 (0.3562)	Prec 93.000% (92.118%)
Test: [60/100]	Time 0.015 (0.023)	Loss 0.3289 (0.3514)	Prec 93.000% (92.180%)
Test: [70/100]	Time 0.015 (0.022)	Loss 0.6297 (0.3493)	Prec 91.000% (92.239%)
Test: [80/100]	Time 0.017 (0.021)	Loss 0.2049 (0.3496)	Prec 94.000% (92.136%)
Test: [90/100]	Time 0.016 (0.020)	Loss 0.3089 (0.3428)	Prec 94.000% (92.220%)
 * Prec 92.260%
```

We provide 5-bit, 3-bit and 2-bit quantized ResNet-20 here, `--bit 5` can be set to 3 and 2.

## ImageNet

We provide 5-bit quantized and 3-bit quantized ResNet-18 here, they can be downloaded in [this link](https://gofile.io/?c=GkchRp), then put them into the checkpoint folder in ImageNet. 

To start validation, change to the ImageNet directory and then: 

```bash
python test.py --bit 3 
```

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

