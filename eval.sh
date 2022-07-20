#!/bin/bash



python eval_timm.py /scratch1/ros282/c3_testset/vis --wandb-run nrosa/marco-2/27d415zm --ckpt test.pth
# python eval_timm.py /scratch1/ros282/c3_testset/vis --wandb-run nrosa/marco-2/1ag02sd9




# python eval_timm.py /scratch1/ros282/marco_full/val \
# --model resnetv2_50 --initial-checkpoint /scratch1/ros282/results/marco_retrain/3139838/marco-2/last.pth.tar \
# --num-classes 4 --input-size 3 256 256 --crop-pct 1 \
# -b 128 --interpolation bicubic

# python eval_timm.py /scratch1/ros282/c3_testset/vis \
# --model swin_tiny_patch4_window7_224 --initial-checkpoint /scratch1/ros282/results/marco_retrain/1155848/marco-2/last.pth.tar \
# --num-classes 4 --input-size 3 224 224 --crop-pct 1 \
# -b 128 --interpolation bicubic

# python eval_timm.py /scratch1/ros282/marco_full/val \
# --model swin_tiny_patch4_window7_224 --initial-checkpoint /scratch1/ros282/results/marco_retrain/1155848/marco-2/last.pth.tar \
# --num-classes 4 --input-size 3 224 224 --crop-pct 1 \
# -b 128 --interpolation bicubic


# python eval_timm.py /scratch1/ros282/c3_testset/vis \
# --model resnet18 --initial-checkpoint /scratch1/ros282/results/marco_retrain/3042111/marco-2/last.pth.tar \
# --num-classes 4 --input-size 3 256 256 --crop-pct 1 \
# -b 128 --interpolation bicubic

# python eval_timm.py /scratch1/ros282/c3_testset/vis \
# --model resnet18 --initial-checkpoint /scratch1/ros282/results/marco_retrain/3042112/marco-2/last.pth.tar \
# --num-classes 4 --input-size 3 256 256 --crop-pct 1 \
# -b 128 --interpolation bicubic

# python eval_timm.py /scratch1/ros282/c3_testset/vis \
# --model resnet18 --initial-checkpoint /scratch1/ros282/results/marco_retrain/3042113/marco-2/last.pth.tar \
# --num-classes 4 --input-size 3 256 256 --crop-pct 1 \
# -b 128 --interpolation bicubic

# python eval_timm.py /scratch1/ros282/c3_testset/vis \
# --model resnet18 --initial-checkpoint /scratch1/ros282/results/marco_retrain/1197056/marco-2/checkpoint-36.pth.tar \
# --num-classes 4 --input-size 3 512 512 --crop-pct 1 \
# -b 128 --interpolation bicubic





# python eval_timm.py /scratch1/ros282/marco_full/val \
# --model resnet18 --initial-checkpoint /scratch1/ros282/results/marco_retrain/1197056/marco-2/checkpoint-36.pth.tar \
# --num-classes 4 --input-size 3 512 512 --crop-pct 1 \
# -b 128 --interpolation bicubic


