# Method 1 - CUB - iNaturalist

We provide a [Jupyter notebook](https://github.com/anguyen8/visual-correspondence-XAI/blob/main/ResNet-50/CUB-iNaturalist/ResNet-Naturalist.ipynb) to reproduce the results for Method 1 (ResNet-50) on the bird classification problem (CUB Dataset). We used a ResNet-50 network pretrained on the iNaturalist dataset as a feature extractor, and we fine-tuned it for the 200-way bird classification problem. For doing so, we use the layer-4 output of the network with the dimension of `2048x7x7` and added an average pool layer, followed by a simple linear layer for classification. We trained the linear layer using the Adam optimizer with all intermediate weights frozen. The resulting network achieves an 85.83% accuracy on the CUB test dataset.

## Download the weights

You can download the weights from [Google Drive](https://drive.google.com/file/d/12jQBvGXXwmh2-HobYZP1mehtMnYgo5pc)