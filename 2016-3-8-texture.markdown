---
layout: post
title:  "Texure, Image Synthesis"
date:   2016-03-8 20:33:35 -0500
---

### Texture Synthesis Using Convolutional Neural Networks
Use pretrained VGG network as a feature embedding method. Given target image, extract features, and compute Gram matrix $$G$$ for each layer $$l$$. The entry $$G_{ij}$$ is the inner product of the vectorized feature map $$F_i$$ and $$F_j$$. The Gram matrix does not have spatial information, because texture, by definition, is spatial invariant, and the measurement does not need spatial information.

The experiments show that the model can generate better image if using the feature map up to pool4 layer of the VGG network.

The algorithm may fail when the target texture image is man made, for example, brick walls.

The Gram matrix, as a new set of features, can also be used for object recognition, a supervised learning task. When using the higher layer of Gram matrix, we can achieve good classification of the objects in the image. This is a bit of surprising, since Gram matrix as a texture feature, does not have any spatial information. But such good classification performance is also consistent with the fact that convnet is also spatial agnostic. (LW: this is probably the reason that some garbage input image can fool convnet even it does not have spatial information, because convnet does not care spatial information)

Interestingly, the authors claim that both "Spatial pyramid pooling in deep convolutional networks for visual recognition" and "Deep convolutional filter banks for texture recognition and segmentation" use similar concept: compute a pooled statistics/features in a stationary feature space.

### A parametric texture model based on joint statistics of complex wavelet coefficients

An early paper that propose a function $$\phi(x)$$ to transform a image to some features, such that image with same textures have similar $$\phi(x)$$ at feature space. This is the original paper that the above texture synthesis work is based on, particular the Gram matrix. Other than the Gram matrix, this paper also propose some other measures, but may not be applicable in the CNN settings, hence not used in the CNN texture synthesis.

That remind me a question that what convolutional net learns. Does it just learn the texture information, which is often sufficient for classification task? I suspect it mainly learns spatial agnostic local statistics, because the full convolutional layer.

Besides the cross-correlation between the feature maps within a layer/scale, this paper also proposes to use cross-correlation between feature maps across layers. Since different scales have different feature map size, finer scale's feature map need to be down-sampled before cross-correlation. however, such texture representation is not used in [gat15].

### Deep Filter Banks for Texture Recognition and Segmentation
Use two CNN for texture classification and recognition. One use the second to last layer of pretrained CNN, so the fully connected layer has spatial information of the object. The other one use the last convolutional layer without the fully connected layer, so it does not have spatial information, and better represent texture information. In practice, the second feature embedding method extract features from multiple scale/layers, just like SIFT.

First generate regions, then do texture classification within regions. Both CNN models are used in the texture classification stage, not in the fist region generation stage. This is because stage one only have a coarse region proposal.

### Describing textures in the wild
This paper is similar to "Deep Filter Bank..." but uses SIFT + Fisher vector, instead of using CNN + FV.

### Bilinear CNN Models for Fine-grained Visual Recognition
Same author has some works on texture representation. And the bilinear CNN is similar to the cross-correlation methods for texture synthesis.

### Separate visual pathways for perception and action
Referenced by the above bilinear CNN. Visual pathway has two path: one for 'where' and another for 'what'. This is yet another paper to show the connection to biological vision.

### Separating style and content with bilinear models
Seem like an early work of bilinear models.  also referenced by bilinear CNN paper.

### Visualizing and Understanding Deep Texture Representations
The same authors of bilinear CNN use the model to inverse network and generate images.

### Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks

### Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis

Looks like the original paper used by Neural-Doodle.

### Texture Networks: Feed-forward Synthesis of Textures and Stylized Images
Latest from Twitter.

### Texture Modeling with Convolutional Spike-and-Slab RBMs and Deep Extensions
A paper I found earlier this year, from bengio's group.

### https://github.com/alexjc/neural-doodle

### https://nucl.ai/blog/extreme-style-machines/

### Neural Autoregressive Distribution Estimation

### Improving the Neural Algorithm of Artistic Style （2016 latest）

### Image Decomposition: Separation of Texture from Piecewise Smooth Content
Google and found it for texture and content separation. Donoho's paper. Note the citation and need to dig into those cited it.  
