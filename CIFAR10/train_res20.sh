#!/usr/bin/env bash
python main.py --arch res20 --bit 32 -id 0,1 --wd 5e-4
python main.py --arch res20 --bit 4 -id 0,1 --wd 1e-4  --lr 4e-2 \
        --init result/res20_32bit/model_best.pth.tar
python main.py --arch res20 --bit 3 -id 0,1 --wd 1e-4  --lr 4e-2 \
        --init result/res20_4bit/model_best.pth.tar
python main.py --arch res20 --bit 2 -id 0,1 --wd 3e-5  --lr 4e-2 \
        --init result/res20_3bit/model_best.pth.tar