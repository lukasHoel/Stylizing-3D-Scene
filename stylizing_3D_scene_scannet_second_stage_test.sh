#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python ddp_test_nerf.py --config configs/test_scene0291_00_second_slurm.txt --render_splits test --style_ID 0
