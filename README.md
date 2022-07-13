<div align="center">    
 
# Visual correspondence-based explanations improve AI robustness and human-AI team accuracy 
by [Giang Nguyen*](https://giangnguyen2412.github.io/), [Mohammad R. Taesiri*](https://taesiri.com/) and [Anh Nguyen](https://anhnguyen.me/). * denotes equal contribution.

[![Paper](http://img.shields.io/badge/paper-arxiv.TBD-B31B1B.svg)]()
[![Conference](http://img.shields.io/badge/XAI4CV@CVPR-2022-4b44ce.svg)](https://xai4cv.github.io/workshop)
[![Conference](http://img.shields.io/badge/poster-4b44ce.svg)](https://www.dropbox.com/s/1neko0pjbexlsjf/p49.pdf?dl=0)
</div> 

_**tldr:** We propose two architectures of interpretable image classifiers that first explain, and then predict by harnessing the visual correspondences between a query image and exemplars.
Our models improve on several out-of-distribution (OOD) ImageNet datasets while achieving competitive performance on ImageNet than the black-box baselines (e.g. ImageNet-pretrained ResNet-50). 
On a large-scale human study (∼60 users per method per dataset) on ImageNet and CUB, our correspondence-based explanations led to human-alone image classification accuracy and human-AI team accuracy that are consistently better than that of kNN. 
We show that it is possible to achieve complementary human-AI team accuracy (i.e., that is higher than either AI-alone or human-alone), on ImageNet and CUB._   

If you use this software, please consider citing:

    @article{correspondence2022,
      title={Visual correspondence-based explanations improve AI robustness and human-AI team accuracy},
      author={Giang Nguyen, Mohammad Reza Taesiri, Anh Nguyen},
      booktitle={XAI4CV workshop at CVPR},
      year={2022}
    }

## 1. Requirements
```
python=3.9.7
pytorch=1.10.0
torchvision=0.11.1
numpy=1.21.2
pip install tqdm
pip install seaborn
```
Since each classifier was developed independently, you can also run ```conda env create -f [classifier_name]/env.yml``` to create the virtual environment before running.

## 2. How to run
### 2.1 Reproduce results
```bash
cd [classifier_name]
sh test.sh
```
### 2.2. Reproduce result on a specific dataset
```bash
cd [classifier_name]
sh run ?  # To see available options
```
Example: Running ```EMD-Corr``` for **CUB-200** using iNaturalist-pretrained ResNet-50 feature.
```bash
sh run.sh -d cub200_test -v False -i True
```

### 2.3. Visualize correspondence-based explanations
### 2.3.1. Visualize EMD-Corr explanations
Example #1, running on a CUB-200 bird image:
```bash
cd EMD-Corr/
# Now turn the visualization (v) option to True
sh run.sh -d cub200_test -v True -i True
```
![](figs/Painted_Bunting_0004_16641.jpeg)
The first row is the exemplars retrieved by kNN while the second one shows the re-ranked exemplars by EMD-Corr classifier.
The third row (query) and the fourth row (EMD-Corr exemplars) show how the query matches each of the exemplars given by comparing the same-color boxes (e.g. face to face, neck to neck).
Both kNN and EMD-Corr correctly classified the bird.

Example #2, running on a general image from ImageNet validation set:
```bash
cd EMD-Corr/
# Now turn the visualization (v) option to True and use torchvision ImageNet-pretrained ResNet-50 feature
sh run.sh -d imagenet1k-val -v True -i False
```
![](figs/ILSVRC2012_val_00003158.jpeg)

### 2.3.2. Visualize CHM-Corr explanations

### 2.4. Try it yourself
#### 2.4.1. Run on your custom dataset
* Step 1: Run kNN to get the shortlisted exemplars. 
* Step 2: Re-rank the exemplars using **EMD** patch-wise comparison.
```python
# EMD-Corr distance calculation between query vs. K exemplars
# fb = fb_center = tensor containing query (first row) + K exemplar embeddings. e.g. 51x2048x7x7 where conv4_dim=2048x7x7.
# q2q_att and g2q_att is the correlation maps of the query (q) to the exemplars (g) or vice versa. 
# opt_plan is the optimal transport plan used for visualization.
from emd_utils import EMDFunctions
emd_distance, q2g_att, g2q_att, opt_plan = EMDFunctions.compute_emd_distance(K=50, fb_center, fb, use_uniform=False, num_patch=5)
```
or 
* Re-rank the exemplars using **CHM** patch-wise comparison.
```python
# CHM-Corr distance calculation between query vs. K exemplars
```
* Step 3: Perform majority vote on the re-ranked exemplars to get the top-1 label.

## 3. Human study
This [video](https://youtu.be/rJx-vGJBprw) walks you through the human study interface. We hope sharing all screens we carefully designed could help future research.
If you wanna try out the UI on your device OR get more materials to replicate the experiment pipeline, hit me up at **nguyengiangbkhn@gmail.com**.

We also share the [training screens](https://drive.google.com/drive/folders/1S0ipBx8H8JDM-tERImHVHFz-YDwE2gf6?usp=sharing) and [test trials](https://drive.google.com/drive/folders/1EWC3hgivx1SA0V2bL2toBnNZvtJWnoGu?usp=sharing) for both human studies on Google Drive.


## 4. License
[MIT](LICENSE)
