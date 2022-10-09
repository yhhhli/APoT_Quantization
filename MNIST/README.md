Here, APoT quantization technique is applied while training ResNet models on the <b>MNIST</b> dataset

### To train ResNet-18 model on 5 bit precision for 25 epochs
```
python main.py --arch resnet18 --bit 5 --epochs 25
```

|   Model   | Precision | Hyper-Params                          | Accuracy |
| :-------: | --------- | ------------------------------------- | -------- |
| ResNet-18 | 5-bit     | batch128_lr0.01_wd0.0001_25epoch      |  99.58   |
| ResNet-18 | 4-bit     | batch128_lr0.01_wd0.0001_25epoch      |  99.49   |
| ResNet-18 | 3-bit     | batch128_lr0.01_wd0.0001_25epoch      |  99.30   |
| ResNet-18 | 2-bit     | batch128_lr0.01_wd0.0001_25epoch      |  98.45   |
| ResNet-10 | 5-bit     | batch128_lr0.01_wd0.0001_25epoch      |  99.78   |
| ResNet-10 | 4-bit     | batch128_lr0.01_wd0.0001_25epoch      |  99.64   |
| ResNet-10 | 3-bit     | batch128_lr0.01_wd0.0001_25epoch      |  99.39   |
| ResNet-10 | 2-bit     | batch128_lr0.01_wd0.0001_25epoch      |  98.72   |


<br/>

### Compared with Uniform Quantization

Pass the value `0` to the argument `power` to switch to uniform quantization
```
python main.py --arch resnet18 --bit 5 --epochs 25 --power 0
```
### Results:

|   Model   | Precision | Hyper-Params                      | Accuracy | Compared with APoT |
| :-------: | --------- | --------------------------------- | -------- | ------------------ |
| ResNet-18 | 5-bit     | batch128_lr0.01_wd0.0001_25epoch  |  99.40   |       -0.18        |
| ResNet-18 | 4-bit     | batch128_lr0.01_wd0.0001_25epoch  |  99.39   |       -0.10        |
| ResNet-18 | 3-bit     | batch128_lr0.01_wd0.0001_25epoch  |  99.24   |       -0.06        |
| ResNet-18 | 2-bit     | batch128_lr0.01_wd0.0001_25epoch  |  98.31   |       -0.14        |
| ResNet-10 | 5-bit     | batch128_lr0.01_wd0.0001_25epoch  |  99.68   |       -0.10        |
| ResNet-10 | 4-bit     | batch128_lr0.01_wd0.0001_25epoch  |  99.54   |       -0.10        |
| ResNet-10 | 3-bit     | batch128_lr0.01_wd0.0001_25epoch  |  99.25   |       -0.14        |
| ResNet-10 | 2-bit     | batch128_lr0.01_wd0.0001_25epoch  |  98.45   |       -0.27        |

### To evaluate the models, you can run:
```
python main.py --arch resnet 18 -e --pretrained result/resnet18_4bit/model_best.pth.tar --bit 4
```
