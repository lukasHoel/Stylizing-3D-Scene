#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python ddp_train_nerf.py --config configs/train_scene0291_00_first_slurm.txt
