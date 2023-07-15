#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
  --train 'F:\Pycharm Projects\pytorch-sepconv\db' \
  --out_dir './output'

python train.py \  --train 'F:\Pycharm Projects\pytorch-sepconv\db' \  --out_dir './output'