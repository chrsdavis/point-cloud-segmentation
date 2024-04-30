# EECS 442 Computer Vision 
### Final Project: Point Cloud Semantic Segmentation with SalsaNext
##### Group members: Alvin Li, Christopher Davis, Fardin Hussain, Keqian Wang


## Reference

Github: [TiagoCortinhal/SalsaNext](https://github.com/TiagoCortinhal/SalsaNext), which is the repo for [SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving](https://arxiv.org/abs/2003.03653) CVPR 2020.

Github: [unmannedlab/RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D), which is the repo for [RELLIS-3D Dataset: Data, Benchmarks and Analysis](https://arxiv.org/abs/2011.12954) CVPR 2020.


## Abstract
This paper discusses and rebuilds the existing SalsaNext for semantic segmentation of a full 3D LiDAR point cloud. SalsaNext is the next version of SalsaNet which has an encoder-decoder architecture where the encoder unit has a set of ResNet blocks and the decoder part combines upsampled features from the residual blocks. To improve from SalsaNext, it introduces a new context module, replaces the ResNet encoder blocks with a new residual dilated convolution stack with gradually increasing receptive fields, and adds the pixel-shuffle layer in the decoder. We also implement the improvement on the Jaccard index by switching from stride convolution to average pooling, applying central dropout treatment, and combining the weighted cross entropy loss with Lov ́asz-Softmax loss.
