# CHM-Corr Classifier

The code and the data for running CHM-Corr classifier. CHM-Corr is a multi-stage classifier; In the first stage, we need to calculate the kNN scores. We provide a GPU-based script to calculate the kNN ranking for images in a validation set. After having kNN scores, we can run the CHM-Corr classifier to get the re-ranked results. The current implementation of CHM-Corr is optimized for a single query, and it will output a pickle file containing all the information about that query, which also can be used for visualization.


## Reproduce results from the paper

### Get kNN Scores

Since, on some GPUs, it is impossible to fit all the validation set scores in the memory, we provide a `split` flag to divide the validation set into `N` subsets automatically. At the end of the process, the code will merge all the results and create a single file for the entire dataset. In our tests, we use split size of `8` on a `A100` gpu.

```bash
python src/kNN/kNN.py --split 10 --name ImageNet --train ~/dataset/ImageNet/train/ --val ~/dataset/ImageNet/val/ --out scores/
```

For convience, we also release the kNN ranking for all the datasets. To download kNN rankings for all datasets, please view `scores/Download.md` or use `scores/Download.py`.

### Run CHM-Corr classifier

There are two CHM-Corr classifiers: A general `CHMCorr.py` for all datasets and `CHMCorr-CUB.py`, which handles the CUB dataset and supports a user-provided mask (Referred to as `CHM-Corr+` in the paper).

For ImageNet validation sets:

```bash
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
```

For the CUB dataset
```bash
# Using iNat Backbone
python src/classifier/CHMCorr-CUB.py --train ~/dataset/CUB_200_2011/train/ --val ~/dataset/CUB_200_2011/test/ --out ~/output/CUB-iNat/ --knn scores/CUB-iNaturalist.pickle  --model inat
# Using ResNet-50 Backbone
python src/classifier/CHMCorr-CUB.py --train ~/dataset/CUB_200_2011/train/ --val ~/dataset/CUB_200_2011/test/ --out ~/output/CUB-ResNet/ --knn scores/scores/CUB-ResNet-50.pickle  --model resnet50
# CHM-Corr+ : Providing a mask 
python src/classifier/CHMCorr-CUB.py --train ~/dataset/CUB_200_2011/train/ --val ~/dataset/CUB_200_2011/test/ --out ~/output/CUB-iNat-Masked/ --knn scores/CUB-iNaturalist.pickle  --model inat --mask masks/CUB-Mask-Top5.pkl
```

## Visualizing the output

We provide a notebook `src/visualization/visualization.ipynb` to visualize the output of the CHM-Corr classifier. After running the CHM-Corr classifier, it outputs a pickle file for each query image; having this pickle file, you can make the visualization.