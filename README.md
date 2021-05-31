# Cross-Domain-Classification

Implementation of domain adaptation on the PACS dataset using pretrained RESNET18 as a feature extractor.

[Colaboratory link](https://colab.research.google.com/drive/1TKiHBl4NXSxLvBL9QvGpdgSrs4YmB0zc?usp=sharing)

## Observations
These observations were made with the photo data as the source domain and the sketch data as the target domain.

- Hyperparameters used for loss function - λ1 = 0.5, λ2 = 0.2, λ3 = 0.3
- A higher weight (λ1) gave good results both in the source domain and target domain
- Increasing the weights (λ2, λ3) led to bad results on the target domain
- SGD with a decreasing learning rate gave the best performance

## Discussion

- The threshold (0.95) on the weighted cross entropy loss could be a tunable hyperparameter
- The source dataset has 1670 images and the target dataset has 3929 images. Additional data augmentation on the dataset might improve the performance of the model
