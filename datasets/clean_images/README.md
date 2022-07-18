# ImageNetClean dataset

This refers to Sec 2.3.2 of the paper: **Visual correspondence-based explanations improve AI
robustness and human-AI team accuracy**.

After filtering out ImageNet validation images:

- having no labels based on ImageNetReaL labels.

- having duplicates with the ImageNet training set.

- having grayscale color or  low resolution (i.e. H or W < 224).

We have 44,424 resultant clean images. We sample queries from these 44K images for human studies.
IDs of these 44K images are in ImageNet_Clean.txt
