---
layout: post
title:  "Semantic Segmentation, Edge Detection"
date:   2016-02-10 20:33:35 -0500
categories: jekyll update
---

Now it's time to have a dedicated page for segmentation.

### Fully Convolutional Networks for Semantic Segmentation

### Hypercolumns for Object Segmentation and Fine-grained Localization
From J. Malik lab, so should read it. Use features at all scales for segmentation. many other tricks, too.

The author thought about possibility of defining a classifier at each pixel, because features do not have spatial information. But to define a location specific classifier for each of 50x50 pixel is too many parameters. So they define a sparser set of classifiers and interpolate to get all pixels. That remind me the U-net paper that just use last 1x1 convolutional layer as classifier, and that amounts to have a classifier at each location?

When using features at all scales of CNN, we get a feature matrix of $$N\times P$$, where $$N$$ is number of pixels and $$P$$ is total number of features across all scales. Then how about a SVD or a dictionary learning over this feature matrix?

Up-sampling the top layers so the feature map has same resolution with previous layers.

This is a well-written paper. Very readable.

### Learning Deconvolution Network for Semantic Segmentation
Yet another segmentation paper.

### Weakly- and Semi-Supervised Learning of a Deep Convolutional Network for Semantic Image Segmentation
This one also cites the Fully convnet paper. Particular interesting because it's weakly supervised.

### Fully Connected Deep Structured Networks
Image segmentation, too. But have a unified learning framework including the feature learning and segmentation.

### Mapping Stacked Decision Forests to Deep and Sparse Convolutional Neural Networks for Semantic Segmentation
Found it because it cite "Deep neural networks segment neuronal membranes in electron microscopy images", and it also cite the "U-Net" paper.

Exact mapping from random forest to CNN ,which can be used to initialize CNN with sparse training data. Approximate mapping from CNN to RF, and the RF improve original RF.

### DeepEdge: A Multi-Scale Bifurcated Deep Network for Top-Down Contour Detection
Shi Jianbo's second author. Also check out recent Arxiv paper for edge detection.

### Conditional random fields as recurrent neural networks
Reformulate mean field approximation by recurrent network. that seems a reminder that old problem (CRF, graph-cut etc) can be re-formulated into a neural network optimization problem, or be re-formulated such that it can be optimized by gradient together with the deep network. Need to read this work.

### Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials

The dense CRF paper. Define unary potential is defined from conventional features like color, texture or spatial location, and the pairwise potentials are define similar to Boykov: a indicator function for the equality of labels, and a mixture of Gaussian kernels as function of features. Here the neighborhood is not local any more, but includes the whole image field. Naive variational inference need update each pixel with cost of O(N). This paper used an approximation: down sampling image, convolution with kernels, and then upsampling. Also used another approximation called permutohedral lattice so the computation is linear with dimension d.

Code is available on author's website with python wrapper (which seems have some functions for > 2D images. )

### Region-based Convolutional Networks for Accurate Object Detection and Segmentation

### PUSHING THE BOUNDARIES OF BOUNDARY DETECTION USING DEEP LEARNING

### ScribbleSup/ Scribble-Supervised Convolutional Networks for Semantic Segmentation
This paper address the problem of segmentation given only scribble about the interior of regions. The problem is similar to Boykov's graph cut, and the optimization also include graph-cut. The unary potential includes two parts: one is from the scribble, and another is from the trained CNN. The pairwise potential is defined like Boykov. The graph is defined on superpixels. The optimization alternates between two steps: 1) Given CNN fixed, optimize graph cut problem on superpixels. 2) Given the segmentation labels of the superpixels estimated from graph-cut, train a fully CNN on each pixel.

### SEMANTIC IMAGE SEGMENTATION WITH DEEP CONVOLUTIONAL NETS AND FULLY CONNECTED CRFS
One of the major CNN+CRF papers.

### Textonboost for image understanding: Multi-class object recognition and segmentation by jointly modeling texture

Seems a major work before deep learning, for feature engineering and detection.

### Multiscale conditional random fields for image labeling

### Higher Order Conditional Random Fields in Deep Neural Networks
Currently the PASCSL VOC leaderboard top. Use potentials that include > 2 nodes. 1) use a object detector and use the detection results (foreground after grabcut) as potentials and 2) use superpixel as higher order potential.

Used mean field update, with fully connected CRF. But the GPU algorithm is not clear to me.

### Fully connected deep structured networks

### Generative Image Modeling using Style and Structure Adversarial Networks

### Understanding Deep Neural Networks by Synthetically Generating the Preferred Stimuli for Each of Their Neurons (paper to come)

### Iterative Instance Segmentation

### http://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/

### Exploiting Local Structures with the Kronecker Layer in Convolutional Networks

### Speeding up convolutional neural networks with low rank expansions (2014)

### 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
