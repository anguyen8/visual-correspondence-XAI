#!/bin/sh

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Running experiments with no visualization (v) and not iNaturalist features (i)
# For ImageNet-val, change self.IMAGENET_REAL in params.py to True to test with ImageNetReaL labels.
sh run.sh -d imagenet1k-val -i False
sh run.sh -d imagenet-r -i False
sh run.sh -d imagenet-sketch -i False
sh run.sh -d DAmageNet_processed -i False
sh run.sh -d adversarial_patches -i False
sh run.sh -d cub200_test -i False
# Using iNaturalist features for CUB200
sh run.sh -d cub200_test -i True
# Small improvements on ImageNet-A, ObjectNet, and ImageNet-C: Not reported in the main text but appendix
sh run.sh -d imagenet-a -i False
sh run.sh -d objectnet-5k -i False
sh run.sh -d gaussian_noise -i False
sh run.sh -d gaussian_blur -i False

