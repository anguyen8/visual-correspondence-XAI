#!/bin/sh

CUDA_VISIBLE_DEVICES=0

# ImageNet
python src/classifier/CHMCorr.py  --train ~/dataset/ImageNet/train/             --val ~/dataset/ImageNet/val/ --out ~/output/ImageNet --knn scores/ImageNet.pickle 
# ImageNet-R (See Symlink.ipynb)
python src/classifier/CHMCorr.py  --train ~/dataset/symlink_parent/ImageNet-R/  --val ~/dataset/ImageNet-R/ --out ~/output/ImageNet-R --knn scores/ImageNet-R.pickle 
# ImageNet-Sketch
python src/classifier/CHMCorr.py  --train ~/dataset/ImageNet-Sketch/            --val ~/dataset/ImageNet-Sketch/ --out ~/output/ImageNet-Sketch --knn scores/ImageNet-Sketch.pickle 
# DAmageNet (See Symlink.ipynb)
python src/classifier/CHMCorr.py  --train ~/dataset/symlink_parent/DAmageNet/   --val ~/dataset/DAmageNet/ --out ~/output/DAmageNet --knn scores/DAmageNet.pickle  --transform multi
# Adversarials Patch
python src/classifier/CHMCorr.py  --train ~/dataset/ImageNet/train/             --val ~/dataset/Adversarials --1out ~/output/Adversarials --knn scores/Adversarials.pickle  --transform multi

# For the CUB dataset
# Using iNat Backbone
python src/classifier/CHMCorr-CUB.py --train ~/dataset/CUB_200_2011/train/ --val ~/dataset/CUB_200_2011/test/ --out ~/output/CUB-iNat/ --knn scores/CUB-iNaturalist.pickle  --model inat
# Using ResNet-50 Backbone
python src/classifier/CHMCorr-CUB.py --train ~/dataset/CUB_200_2011/train/ --val ~/dataset/CUB_200_2011/test/ --out ~/output/CUB-ResNet/ --knn scores/scores/CUB-ResNet-50.pickle  --model resnet50
# CHM-Corr+ : Providing a mask 
python src/classifier/CHMCorr-CUB.py --train ~/dataset/CUB_200_2011/train/ --val ~/dataset/CUB_200_2011/test/ --out ~/output/CUB-iNat-Masked/ --knn scores/CUB-iNaturalist.pickle  --model inat --mask masks/CUB-Mask-Top5.pkl