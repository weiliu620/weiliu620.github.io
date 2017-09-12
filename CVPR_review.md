---
title: CVPR review
description: A quick review of the papers worth reading in CVPR 2017
mathjax: true
---

## Overall impression

- Multi-modality
- RL can be used in ML/CV problem, too, if we define the problem carefully
- high level image understanding are more popular. Lower level vision is less. IT's OK to have some errors at lower level, and we can use context and other modality information at higher level for better inference.


### Robobarista: Object Part-based Transfer of Manipulation Trajectories from Crowd-sourcing in 3D Pointclouds

Scene understanding and part detection is not sufficient for robot manipulation. This paper learn direct perception base on affordance. [search 'affordance']. Also this paper use 'learning from demonstration', but learn a single action is impossible because environment has many variations.

The goal of the task is mapping a point cloud and a language to a trajectory. First, trajectory should not be represented in configuration space (prone to errors when aligned with object), but in task space, and in principal axis based coordinate frame. The cost function is defined by the dynamic time warping function of the target trajectory and the trajectory to be optimized. DTW is a method to match two time series in a nonlinear way.

### Densely Connected Convolutional Networks

Best paper of CVPR 2017.

### End-to-End Driving in a Realistic Racing Game with Deep ...
A workshop paper. Traditional autonomous driving is perception, planning and control, but recently map sensor input directly to low level control. This work uses A3C for learning to control. Also see [End-to-end training of deep visuomotor policies]. A3C is good because it does not need any experience replay. Reward is not defined by the in-game score since it's too sparse. The reward is defined by the distance to the middle of the road. [LW: seems in practical RL problem, we can define the reward arbitrarily to reflect our prior knowledge].

### Automated risk assessment for scene understanding and domestic robots using RGB-D data and 2.5D CNNs at a patch level

This work seems nothing new but the application is related to XOM business. May need task-specific training dataset.

### The One Hundred Layers Tiramisu/ Fully Convolutional DenseNets for Semantic Segmentation

Extend Dense Network to up-convolution. Works well on some datasets without preprocessing and post-processing.  [Need to implement this in TF]

### Borrowing Treasures From the Wealthy: Deep Transfer Learning Through Selective Joint Fine-Tuning

Oral paper. Source and target task are learned at same time. Use subset of training data of source task, and fine tune the shared layer of both tasks.

### The More You Know: Using Knowledge Graphs for Image Classification

Graph search neural net, use features from the image to annotate the graph, select subset of graph and predict the output on nodes representing visual concepts.

### A contextual constructive approach

use graph neural nets to QSPR in chemistry, but could not find pdf online. Not a CVPR paper.

### Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs

Define convolution on graph structure and take into account the label defined on nodes. [think about connections to CRF or annistropic diffusion ]. Applied to point cloud data, but also useful for other graph structure like chemical, etc. 3D modeling, computational chemistry and biology, geospatial analysis, social networks, or natural language semantics and knowledge bases,

### Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks

Use GAN to generative real images given synthetic images, similar to the GAN paper from Apple, best paper of CVPR 17.

The paper seems well written, at least the oral talk slides are good. [Need to read paper to know more basics of DA]

Feed-forward based model that can generated multiple textures from single network. Represent style with a one-hot vector but continuous. Input is from noise and also from style vectdor. Has a diversity loss

### Deep Reinforcement Learning-based Image Captioning with Embedding Reward ***

Define a policy network to predict the next word, and a value network to evaluate all possible extensions. Existing methods use encoder-decoder: use convnet to encode images, and then use recurrent net to decode it into sentences. recurrent net is greedy, so propose RL for global optimal prediction. Actic-critic net, and use visual-semantic embedding to define the reward.

### Agent-Centric Risk Assessment: Accident Anticipation and Risky Region Localization

Didn't read it but seems related with XOM safety.

### Action-Decision Networks for Visual Tracking with Deep Reinforcement Learning

light computation, deep net is pre-trained, and then online adapted. Pretraining is by RL,

### Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach


### Network Dissection: Quantifying Interpretability of Deep Visual Representations

### MDNet: A Semantically and Visually Interpretable Medical Image Diagnosis Network

### Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs

### Feature Pyramid Networks for Object Detection

### Deep Variation-Structured Reinforcement Learning for Visual Relationship and Attribute Detection

### Learning random-walk label propagation for weakly-supervised semantic segmentation

### Deep Watershed Transform for Instance Segmentation

### Mining Object Parts from CNNs via Active Question-Answering

Zhu Song-Chun's lab.

### PoseAgent: Budget-Constrained 6D Object Pose Estimation via Reinforcement Learning

Use policy network for RL.

### StyleBank: An Explicit Representation for Neural Image Style Transfer

Microsoft Asia Pix paper.

### Joint Sequence Learning and Cross-Modality Convolution for 3D Biomedical Segmentation
GE GRC paper.


### Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes
two networks for segmentation, without pre-training and post-processing.

### Deep Variation-structured Reinforcement Learning for Visual Relationship and Attribute Detection
Use RL to detect something.

### Detecting Visual Relationships with Deep Relational Networks
Another relationship detection work, oral talk.

### A Reinforcement Learning Approach to the View Planning Problem
GE paper.

### RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation

### Low-rank Bilinear Pooling for Fine-Grained Classification
seems loosely connected to textures

### Knowledge Acquisition for Visual Question Answering via Iterative Querying

Zhu Yuke and Fei-Fei's work.

### Cognitive Mapping and Planning for Visual Navigation
Malik's paper. Also saw it on his workshop talk.

### Deep Feature Interpolation for Image Content Changes

### Collaborative Deep Reinforcement Learning for Joint Object Search

### Image-to-Image Translation with Conditional Adversarial Networks

### G2DeNet: Global Gaussian Distribution Embedding Network and Its Application to Visual Recognition

some structure layers is trainable and can be inserted in to DNN. This paper proposes a Gaussian embedding layer, which model the probabilistic Distribution of the filter output. This paper may have some references about other global trainable layers. Decompose the covariance matrix into sub-matrices.

$$x^2 + y^2$$
