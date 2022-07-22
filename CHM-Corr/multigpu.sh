#!/bin/sh

tmux new-session -d -s TESTALL

## Create the windows on which each node or .launch file is going to run

tmux send-keys -t TESTALL 'tmux new-window -n ImageNet ' ENTER
tmux send-keys -t TESTALL 'tmux new-window -n ImageNetR ' ENTER
tmux send-keys -t TESTALL 'tmux new-window -n ImageNetSketch ' ENTER
tmux send-keys -t TESTALL 'tmux new-window -n DAmageNet ' ENTER
tmux send-keys -t TESTALL 'tmux new-window -n Adversarial ' ENTER
tmux send-keys -t TESTALL 'tmux new-window -n CUBiNAT ' ENTER
tmux send-keys -t TESTALL 'tmux new-window -n CUBResNet ' ENTER
tmux send-keys -t TESTALL 'tmux new-window -n CUBiNATMasked ' ENTER

## Send the command to each window from window 0
tmux send-keys -t TESTALL "tmux send-keys -t ImageNet       'conda activate  pytorch1.11; export CUDA_VISIBLE_DEVICES=0; python src/classifier/CHMCorr.py --N 50 --K 20 --T 0.55  --train ~/dataset/ILSVRC2012_img_train/       --val ~/dataset/ILSVRC2012_img_val/ --out ~/output/ImageNet --bs 50 --knn /home/mohammad/src/Manuscript-Results/KNN/ImageNetVAL_50K.pickle' ENTER" ENTER
tmux send-keys -t TESTALL "tmux send-keys -t ImageNetR      'conda activate  pytorch1.11; export CUDA_VISIBLE_DEVICES=1; python src/classifier/CHMCorr.py --N 50 --K 20 --T 0.55  --train ~/dataset/symlinked_imagenet_training/imagenet-r/  --val ~/dataset/imagenet-r/ --out ~/output/ImageNet-R --bs 50 --knn ./scores/ImageNet-R.pickle' ENTER" ENTER
tmux send-keys -t TESTALL "tmux send-keys -t ImageNetSketch 'conda activate  pytorch1.11; export CUDA_VISIBLE_DEVICES=2; python src/classifier/CHMCorr.py --N 50 --K 20 --T 0.55  --train ~/dataset/ILSVRC2012_img_train/       --val ~/dataset/imagenet-sketch/ --out ~/output/ImageNet-Sketch --bs 50 --knn ./scores/ImageNet-Sketch.pickle ' ENTER" ENTER
tmux send-keys -t TESTALL "tmux send-keys -t DAmageNet      'conda activate  pytorch1.11; export CUDA_VISIBLE_DEVICES=3; python src/classifier/CHMCorr.py --N 50 --K 20 --T 0.55  --train ~/dataset/symlinked_imagenet_training/DAmageNet/   --val ~/dataset/DAmageNet/mapped/ --out ~/output/DAmageNet --bs 50 --knn ./scores/DAmageNet.pickle  --transform multi' ENTER" ENTER
tmux send-keys -t TESTALL "tmux send-keys -t Adversarial    'conda activate  pytorch1.11; export CUDA_VISIBLE_DEVICES=4; python src/classifier/CHMCorr.py --N 50 --K 20 --T 0.55  --train ~/dataset/ILSVRC2012_img_train/       --val ~/dataset/adversarial_patches --out ~/output/Adversarials --bs 50 --knn ./scores/Adversarials.pickle  --transform multi' ENTER" ENTER
tmux send-keys -t TESTALL "tmux send-keys -t CUBiNAT        'conda activate  pytorch1.11; export CUDA_VISIBLE_DEVICES=5; python src/classifier/CHMCorr-CUB.py  --N 50 --K 20 --T 0.55  --train ~/dataset/CUB_200_2011/train1/ --val ~/dataset/CUB_200_2011/test0/ --out ~/output/CUB-iNat/ --bs 50 --knn ./scores/CUB-iNaturalist.pickle  --model inat' ENTER" ENTER
tmux send-keys -t TESTALL "tmux send-keys -t CUBResNet      'conda activate  pytorch1.11; export CUDA_VISIBLE_DEVICES=6; python src/classifier/CHMCorr-CUB.py  --N 50 --K 20 --T 0.55  --train ~/dataset/CUB_200_2011/train1/ --val ~/dataset/CUB_200_2011/test0/ --out ~/output/CUB-ResNet/ --bs 50 --knn ./scores/CUB-ResNet-50.pickle  --model resnet50' ENTER" ENTER
tmux send-keys -t TESTALL "tmux send-keys -t CUBiNATMasked  'conda activate  pytorch1.11; export CUDA_VISIBLE_DEVICES=7; python src/classifier/CHMCorr-CUB.py  --N 50 --K 20 --T 0.55  --train ~/dataset/CUB_200_2011/train1/ --val ~/dataset/CUB_200_2011/test0/ --out ~/output/CUB-iNat-Masked/ --bs 50 --knn ./scores/CUB-iNaturalist.pickle  --model inat --mask ./masks/CUB-Mask-Top5.pkl' ENTER" ENTER


## Start a new line on window 0
tmux send-keys -t TESTALL ENTER

## Attach to session
tmux send-keys -t TESTALL "tmux select-window -t Dataset1" ENTER
tmux attach -t TESTALL







