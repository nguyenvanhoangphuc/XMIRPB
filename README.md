# X-MIR: EXplainable Medical Image Retrieval

Code accompanying the WACV '22 paper "X-MIR: EXplainable Medical Image Retrieval"

## Embedding model for image retrieval on chest X-ray images

This repository trains an embedding model using metric learning techniques (e.g. triplet loss), which can then be used for image retrieval on chest X-ray or skin lesion images. The main application is to aid clinicians in diagnosing different cases (e.g. COVID or carcinoma). This repository makes heavy use of other publicly available resources:

### Dataset
[COVIDx dataset](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md)

### Pretrained model
[CheXNet](https://github.com/arnoweng/CheXNet)

### Similarity learning using triplet loss
[Pytorch Reference](https://github.com/pytorch/vision/tree/master/references/similarity)


## Similarity-based saliency maps
This repository contains implementations of different similarity-based saliency maps. The code is inspired by the [RISE](https://github.com/eclique/RISE) repository.

### Implemented Methods
[Explainability for Content-Based Image Retrieval](http://openaccess.thecvf.com/content_CVPRW_2019/html/Explainable_AI/Dong_Explainability_for_Content-Based_Image_Retrieval_CVPRW_2019_paper.html)  
Black-box saliency method based on occlusion sensitivity

[Visualizing Deep Similarity Networks](https://arxiv.org/abs/1901.00536) / [Visual Explanation for Deep Metric Learning](https://arxiv.org/abs/1909.12977)  
White-box saliency method based on pairwise similarity between features in the last convolutional layer

[Learning Similarity Attention](https://arxiv.org/abs/1911.07381)  
White-box saliency method based on image triplets: anchor, positive, and negative examples

We also include code for producing saliency maps (including self-similarity saliency maps) which can then be evaluated using different metrics.