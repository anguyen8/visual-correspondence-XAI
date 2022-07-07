#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3

# Running experiments with no visualization (v) and not iNaturalist features (i)
# For ImageNet-val, change self.IMAGENET_REAL in params.py to True to test with ImageNetReaL labels.
sh run.sh -d imagenet1k-val -v False -i False
sh run.sh -d imagenet-r -v False -i False
sh run.sh -d imagenet-sketch -v False -i False
sh run.sh -d DAmageNet_processed -v False -i False
sh run.sh -d adversarial_patches -v False -i False
sh run.sh -d cub200_test -v False -i False
# Using iNaturalist features for CUB200
sh run.sh -d cub200_test -v False -i True
# Small improvements on ImageNet-A, ObjectNet, and ImageNet-C: Not reported in the main text but appendix
sh run.sh -d imagenet-a -v False -i False
sh run.sh -d objectnet-5k -v False -i False
sh run.sh -d gaussian_noise -v False -i False
sh run.sh -d gaussian_blur -v False -i False

