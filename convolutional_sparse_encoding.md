---
title: Convolutional autoencoder, sparse coding, etc.
---

### What is the best multi-stage architecture for object recognition
Not a convolutional sparse coding, but an earlier work of the same lab. Used patch based model for multi-stage learning (unsupervised?)

Eq. 1 in the paper may help me understand the difference between 2D and 3D convolution.

I vaguely remember this paper also mentioned that some methods use advanced classifiers after feature learning/extraction. This might be a starting point to learn these advanced methods (such as spatial pyramid matching). Haven't paid much attention to this area yet.

### Convolutional matching pursuit and dictionary training

### Deconvolutional Networks (2010)
Zailer's convolutional sparse coding seems the first one to use it, even before "Learning convolutional feature hierarchies...".

### Learning convolutional feature hierarchies for visual recognition (2010)
The author also made the point that conventional sparse coding need a optimization during test, even a dictionary is already learned during training. The proposed methods also use a encoding step to approximate the codes (just like early work of Ranzato's autoencoder)

It seems the methods proposed here need a lot of optimization work. the author claims that the filter learned in convolutional sparse coding is more diverse, with less redundancy. But what is the real benefits of it even it sounds very good? Is it worth more work during learning? I wish there is a Theano or Tensorflow package for this algorithm.


Digress: Early deep network is trained layer by layer, but recent autoencoder seems not. Do we still need train each layer after previous layer is trained?

### Convolutional matching pursuit and dictionary training
Also Zeiler's paper.

### Fast Convolutional Sparse Coding (2013)

### Deep learning with hierarchical convolutional factor analysis (IEEE PAMI 2013)

### Imaging in scattering media using correlation image sensors and sparse convolutional coding(2014)

### Fast and Flexible Convolutional Sparse Coding (2015)
Previous Bristow's method do the convolution in frequency domain, but this paper proposed new optimization method that improve both the spatial and frequency domain learning.

The experiments show that when the input image is big (say, 1000x1000), this method is much faster than patch-based methods.

Local contrast normalization? What is that.

A unrelated question: how to stack multiple SC to get a deep network. Choose filter size? What is the feature map size? Max-pooling or not? May need to go back to LeCun's early paper. (what is the best model structure for deep network...)

### Imaging in scattering media using correlation image sensors and sparse convolutional coding.
Same author of previous paper. Use sparse coding on data with noisy and missing measurements, when filters are know as a priori (for example, motivated by physical models, such as sonar, seismic imaging, radar or ultrasound). However, the toy example shown in the paper is not very convincing.

### From sparse solutions of systems of equations to sparse modeling of signals and images
A patch based method.

### Online dictionary learning for sparse coding
A patch based method

### From learning models of natural
image patches to whole image restoration
Patched based.

### Shift-invariance sparse coding for audio classification (UAI 2007)
First to propose solve sparse coding in frequency domain.

### Fast Convolutional SparseCoding (FCSC)
Technical report in 2014. Sove SC in frequency domain for 2D image. but may have boundary artifacts.

### Stacked what-where auto-encoders (2015)

### Brendt Wohlberg, "Efficient Convolutional Sparse Coding"
Has a matlab code repository on GitHub. Also cited Bristow.

### Classification of Histology Sections via Multispectral Convolutional Sparse Coding (2014, CVPR)
Cited Bristow's work. CVPR paper. (remind me the EDS multiple channel dataset...can be useful for this purpose, or multi-modality seismic data)

Learned features are sent into a spatial pyramid matching for classification. It may work better than vanilla classifier.

The paper looks like an good application of CS in biological imaging.

### Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories.
Spatial pyramid matching.

### Character- ization of tissue histopathology via predictive sparse decom- position and spatial pyramid matching (2013)

### Pedestrian detection with unsupervised multi-stage feature learning
LeCunn's group. use sparse coding.

### Accurate and efficient linear structure segmentation by leveraging ad hoc features with learned filters
On retinal segmentation.
--------------------------------------------
