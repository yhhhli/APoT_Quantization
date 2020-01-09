#!/usr/bin/env bash
python main.py --arch res56 --bit 32 -id 2,3 --wd 5e-4
python main.py --arch res56 --bit 4 -id 2,3 --wd 1e-4 --lr 4e-2 \
        --init result/res56_32bit/model_best.pth.tar
python main.py --arch res56 --bit 3 -id 2,3 --wd 1e-4 --lr 4e-2 \
        --init result/res56_4bit/model_best.pth.tar
python main.py --arch res56 --bit 2 -id 2,3 --wd 3e-5 --lr 4e-2 \
        --init result/res56_3bit/model_best.pth.tar