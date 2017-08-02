---
title: medical imaging
---

### Large-Scale Automatic Reconstruction of Neuronal Processes from Electron Microscopy Images

### Flexible, high performance convolutional neural networks for image classification (IJCAI, 2011)

### Deep neural networks segment neuronal membranes in electron microscopy images
An early 2012 work on cell membrane segmentation. Used patched based feature embedding methods and standard convnet. Winner of 2012 ISBI challenge.

- The positive samples are selected as the square image centered at pixels labeled as membrane. Negative samples are selected at rest of the pixel labeled as non-membrane, with same number of positive samples. Because of the rotation invariance, data is augmented at beginning of each epoch by rotating 90 degree and mirroring.

- Training data has equal number of positive and negative samples, so the output scores are biased. Trained another polynomial transformation to correct such bias.

- For such supervised learning, it seems there is no difference between so called patch-based or image based. During training, a whole image enter the convnet, and cost function is defined based on the segmentation labels.

- For unsupervised learning such as autoencoder, there might be some difference. Need revisit this difference since I forgot it.  
- Also manipulated input data window by blurring peripheral pixels and fisheye'ing the image.
- Also used model ensembles: multiple CNNs.

### Mitosis Detection in Breast Cancer Histology Images with Deep Neural Networks
Also patch based methods and use max-pooling. I think patch based methods make sense for these microscopy images, because there is not much hierarchical information. Most of the interesting patterns are at roughly same scale. That is, looking only at at patch does not lose any information for the recognition of objects within the patch.

### Multi-column deep neural networks for image classification (2012)
An early work that are well cited. Use multiple DNN as a ensemble method.

### Deep Neural Networks for Object Detection (2013)
This is detection, not segmentation. and not really medical imaging domain.

### Fast Image Scanning with Deep Max-Pooling Convolutional Neural Networks
A technical report of 2013 for segmentation, also on microscopy images. This paper seems to compare patch-based and image-based CNN. May worth reading.

Instead of computing convolution between each patch and filter, we can compute the convolution between whole input image and a filter. However, max-pooling make it tricky. This paper deal with convolution layer and max-pooling layer, and still do a image-based convolution, instead of a patch-based. Could not figure out how the "fragment" trick works...

### SegEM: Efficient Image Analysis for High-Resolution Connectomics
Not a deep learning paper per se, but an image analysis paper on Neuron.

### Automatic Detection of Cerebral Microbleeds from MR Images via 3D Convolutional Neural Networks.
Found this one searching vonvnets on 3D volume imaging.

### A fast and scalable algorithm for training 3D convolutional networks on multi-core and many-core shared memory machines
Seems more on computational speed up, but also like application on 3D volume image. From the group for neuron membrane study (image segmentation?)

First pass of the paper: ZNN works better on networks with larger kernel/filter size. For 3D convolution, ZNN starts to work faster when kernel is 7x7x7.

## Deep Learning Convolutional Networks for Multiphoton Microscopy Vasculature Segmentation
Found this paper since it cited the above ZNN paper. It's about vessel segmentation in 3D. A dataset of 12 volume is available for public. (has some reproducible research references that are quite interesting). Author finland.

### Recursive Training of 2D-3D Convolutional Networks for Neuronal Boundary Detection
An application of the ZNN paper above to the 3D volume imaging, for neuron reconstruction problem in Seung's group. Their network seems to include 2D convolution filters at beginning, then 3D filter at later stage. Probably it is because of the anisotropic resolution of EM image.

The authors also mentioned the 3D convnet applications on block face EM data (isotropic). Need to check out these work.

The paper also cited two works of using 3D filters on video processing. That reminds me that video analysis are also applications of 3D convnet.

The paper also use a max-filtering instead of max-pooling. The output of max-filtering has same size with input map. Not sure why it is preferred. Probably for a pixel-wise training.

The recursive method proposed in this work is inspired by a Neuron paper on how Monkeys recognize contours. May need to check out the original work.

### Convolutional networks can learn to generate affinity graphs for image segmentation (2010)
An early paper cited by above paper, about convnet on 3D image segmentation. It is from the Seung's group for neuron segmentation. Use CNN to extract features, and use these features to define the affinity graph. Edge weights is 1 for two points similar, and 0 otherwise.

the CNN model's output is the weight of edges on the graph of 3D grid. Such model makes sense for this task because there is a natural boundary between the objects of interests.

Once the edge weights ( 1 or 0) is learned, partitioning is done by N-Cuts.

### Supervised Learning of Image Restoration with Convolutional Networks
For image restoration with CNN, also compare with MRF and claim they are related, but CNN works better in practice.

###  Crowdsourcing the creation of image segmentation algorithms for connectomics

### [Robust Cell Detection and Segmentation in Histopathological Images using Sparse Reconstruction and Stacked Denoising Autoencoders][su15]
MICCAI paper. Also check out Zhang, Shaoting's other publications, especially on the hashing work.

First pass: looks like many steps in the processing pipeline, and I'm not sure whether autoencoder play a big role here. It is a 2D object detection and recognition problem. Not 3D.

### [Deep convolutional encoder networks for multiple sclerosis lesion segmentation][bro15]
A MICCAI paper. Convolutional net on entire volume image instead on patches. The deconv layer is similar to Zeiler's work. The model is also similar to the convolutional autoencoder by Masci et al. However, the model in this paper is not deep, just 3 layers. on the implementation side, no mention about how to deal with 3D volume image as input. Standard convnet only take 2D image plus RGB channels.

One thing not clear to me: the encoding uses "valid convolution", but the decoding uses "full convolution".

### [Deep learning of image features from unlabeled data for multiple sclerosis lesion segmentation]
From the same Tam's group. Patched based learning. Check the lab's publications [here](https://www.cs.ubc.ca/~rtam/publications.html)

A MICCAI workshop paper. First unsupervised then supervised training. Unsupervised learning is on a stacked RBM, then the trained RBM is used to extract feature for labeled data and send the feature to a voxel-wise random forest for training.

For RBM, patches with two different scales are extracted centered at a voxel. RBM has two hidden layer of binary values. The features extracted from two scales are concatenated, also the features from T2 and PD modalities are concatenated. Extracted some patches with true target (lesion), and also extracted some patches without target pattern.

### Deep 3D convolutional encoder networks with shortcuts for multiscale feature integration applied to multiple sclerosis lesion segmentation
From the same group of previous MICCAI paper. This is a TMI paper. Could not find a online copy now. From the abstract: use convolution and deconvolution for segmentation. Seems have a unsupervised step then a supervised step. Also use shortcuts (between conv and deconv layer?) for what reason?

### Deep learning of brain images and its application to multiple sclerosis
Also from the same group. Well this is a MLMI workshop proceedings so may not worth reading.

### [Spatially-sparse convolutional neural networks](http://arxiv.org/pdf/1409.6070v1.pdf)
Not a medical imaging application, but a trick of convnet for sparse 3D spatial data. For example, a pen stroke in 3D space is sparse. Saw this paper on a reddit discussion for 3D convnet.

### DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation
Multi-atlas and label fusion is top-down, and this paper claim to be bottom-up. Run a super-voxel before convnet, but why?

Also use R-CNN to generate region proposal, at different scales.

Not very clear why so many processing steps are needed. Probably the region proposal will save computation in 3D space. Also, the superpixel and region generations is fairly new so might be helpful for related work.

Extract patches at axial, coronal, and sagittal slice, around each supervoxels.

It seems that a initial convnet generate some segmentation, which is used for a second convnet. Such methods looks familiar.

Also use nonrigid deformation to augment patch data. Deformation use thin plate spline.

At last the probability score from all scales are combined. Overall, it seems there are too many steps and I'm not sure such pipeline is generalizable.  

### 3D Deep Learning for Efficient and Robust Landmark Detection in Volumetric Data
Have read this before. Second read: Problem of 3D volume: more input pixels, and need more training data. Two steps: First step use a shallow network to find candidates, and second step use deep network for classification. The second network looks like unsupervised learning by using stack autoencoder, and the features learned all all layers are sent into other classifiers.

For first network, use separable 3D filter, so 3D convolution is approximated by three 1D convolution. The original 3D filter $$W$$ can be approximated by $$W = \sum_s W_x^s \cdot W_y^s \cdot W_z^s$$. Such sum over multiple terms can have better approximation than a single multiplication. Furthermore, because many filter $$W$$'s in a single layer tends to be strong correlated, we can use same set of 1D filter to reconstruct all 3D filters $$W$$ in a layer. So, we need  a 4D tensor decomposition.

In order to use smaller number of separable filters, they apply constraints on the original 3D filter to have small variance, in order to make them smooth. That remind me the topological smooth filters learned from LeCunn's group, which seems make more sense than simply regularizing the variance of filters.

The second network use MLP and has 3 hidden layers and 2000 nodes each layer. The authors claim that separable filters do not work well in the second network because target objects are scattered around. Why is that? Is that because the filters within layers are less correlated so we cannot approximate the filters only using a small number of 1D filters?

The author observes that many filters' weights are small, so they enforce that by applying a sparsity regularization. And, it is not clear to me that the second network is a autoencoder which means it's unsupservised. So, the features learned in the second networks are used for other classifiers.

The patch size: too small we don't have enough field of view to do correct classification, too large will add computational cost. The authors build image pyramid. It seems they downsampled the image and extract same size patches, that amounts to the large FOV in original scale. But, I'm thinking, isn't it the same thing with multi-layer convolution network? (which the author didn't use, or at least didn't get better results from). The features from multi scales are concatenated into the classifier.

The classifiers: probabilistic boosting tree. That may answer my questions: how to deal with the feature extracted from multi-scale? Just let the classifier do the job.

The experiments seem to show that using both predefined features such as Haar wavelet, and the learned features from deep network achieves much better results. That is a reminder that engineered features can still have complementary information to the learned features.

Overall, I think the two step approach makes sense on detections problems where we have sparse positive samples in a large image area (which is what we're dealing with)

The separable filters are much more efficient than 3D filter only when the filter size is large. However, nowadays filter size tends to be small. Is it still worthy to approximate 3D filter by separable 1D filters?

### Deep feature learning for knee cartilage segmentation using a triplanar convolutional neural network (MICCAI 2013)
Use orthogonal patches as a surrogate to 3D patch and train the model.

### A new 2.5D representation for lymph node detection using random sets of deep convolutional neural network observations (MICCAI 2014)
Extract 2D patches with random orientation.

### A Hybrid of Deep Network and Hidden Markov Model for MCI Identification with Resting-State fMRI
Paper not available online. Use autoeoncer and HMM for classificaiton, and claim to model dynamics.

### Automatic Coronary Calcium Scoring in Cardiac CT Angiography using Convolutional Neural Networks
Original paper not available online, but abstract seems too application specific and not much novelty on the modeling. Will revisit later if I really have time.

### Deep Learning and Structured Prediction for the Segmentation of Mass in Mammograms
Two feature learning methods: deep belief networks, and deep convolutional networks, and two classifiers: CRF, and structured SVM.

Not very readable. Looks like CNN use patch-based supervised training. Theoretically CNN can be used as classifier, too, but the authors claims it overfits. So, the CNN is only used to extract features. Other features from a GMM also used and improves results. Not sure how much role CNN played in the over all performance.

### Learning Tensor-Based Features for Whole-Brain fMRI Classification

### Automatic Diagnosis of Ovarian Carcinomas via Sparse Multi Resolution Tissue Representation
Unsupervised learning, dictionary learning. First, unsupervised learning with three layer of sparse coding of Zeiler's, and Fisher vector to extract features and train a linear SVM.

Extract patches with two sizes. (I've seen same practice before). The experiments also show that Fisher vectors is better than BoW with SPM.

One thing not clear to me: classifier's output is aggregated with geometric mean? It seems the classification is on a whole tissue sample, which includes multiple patches. Each patches get a classification result, and the final result of the tissue sample is aggregated from all patches extracted from the tissue. Right?

Also this paper "The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization" is cited to show the importance of feature coding.

### The devil is in the details: an evaluation of recent feature encoding methods.
Cited by above paper about the Fisher vector embeddings.

<!-- ### Direct and Simultaneous Four-Chamber Volume Estimation with Supervised Feature Learning
Cannot found paper online. Not a deep learning paper. Can be skipped.
 -->
### Cross-Domain Synthesis of Medical Images Using Efficient Location-Sensitive Deep Network

### Marginal Space Deep Learning: Efficient Architecture for Detection in Volumetric Image Data

### U-Net: Convolutional Networks for Biomedical Image Segmentation
ISBI 2015 winner and looks really good paper. Built based on the 'fully convolutional network' of Long et al. Conventional net + upconvolutional (upsampling) net. The features at certain convolutional layer can take the shortcut to the corresponding upconvolutional layer, without going to layer in next scale. the upconvolutional net takes feature both from the shortcut and the up-sampled feature map. Such methods remind me the ladder network proposed recently that achieves good performance only use sparse labeled data.

The authors also referred to "hyper column segmentation" by J. Malik and "Image segmentation with cascaded hierarchical models and logistic disjunctive normal networks" by M Seyed for classification using features from all scales. This paper took a different approach that just define additional special layers (1x1 conv) as a classifier, and learn it together with the rest of CNN. So, there is no need of another classifier. (need to read the 'network in network' paper because the 1x1 conv layer is from that paper)

Difference with the 'full convolutional network': up-sampling also has large number of channels. (need to revisit the fully convnet for details). Also, there is no full connected layer, which makes it possible to take arbitrary input image size. My understanding is the full connected layer is the only constraints on the input image size. Conv layer does not care about input size. So, user can choose input image size based on GPU memory.

Some cells touch with each other and make the segmentation difficult. So redefine weighted loss, so the background pixels between touching cells have more weights.

Not clear: at the end of upconvolutional net, the 1x1 conv layer maps each of 64 components feature vector to the desired number of classes. It looks like this speical 1x1 conv layer is used as a classifier, and the weights are trained together with the rest of the network. This method remind the the ICCV 2015 paper of training convnet and a random decision tree together. Also I've seen somewhere the 1x1 conv layer is used smartly.

The algorithm still use patches, but the patch is much larger than the 2012 ISBI winner. Not necessary to take whole image as input because of possible memory constraint. But such large patch is enough information for segmentation. But how the tiling works? Overlapped patches and segmentation map? I think this is some implementation details that are not that important, compared to the 1x1 conv layer as classifier.

Also use deformation to augment data set.

The author is at DeepMind. Code available online (in Matlab). Also an alternative implementation at [here](http://arxiv.org/abs/1509.03371).

### A Novel Cell Detection Method Using Deep Convolutional Neural Network and Maximum-Weight Independent Set
paper not available online.

### Beyond Classification: Structured Regression For Robust Cell Detection Using Convolutional Neural Network
Another cell detection method, that take care of touching cells. But generate a heat map of each cell such that cell center has higher values. The author claims that Ciresan's membrane detection is pixel based and ignore topology of the map. The model in this paper replace the last layer of conventional CNN by a structure regression layer. To build the model, a target heat-map like mask is defined such that pixels close to true cell center has higher values, and this mask is used as target. Then somehow a cost function is defined (details not clear to me). The remaining is standard CNN from input image 49x49x3.

Each pixel gets prediction from multiple neighbors, so also need a averaging as a fusion step.

Remaining part of the paper is not accessible from Google book.

### Automatic Feature Learning for Glaucoma Detection based on Deep Learning
paper not available to public, but the idea is to use a 'mlpconv' (network in network paper in 2013) in place of convolutional layer. Also take the output of one CNN as a context input of its own full connected layer (not clear what it means)
### Fast Automatic Vertebrae Detection and Localization in Pathological CT Scans - A Deep Learning Approach
Paper not available online. CNN output is the relative position (a vector pointing to the center)

### Unregistered Multiview Mammogram Analysis with Pre-trained Deep Learning Models

### Deep Neural Networks for Anatomical Brain Segmentation (CVPR workshop 2015)
Used both 3D and 2D orthogonal patches, and also use larger 2D patches. The 2D patches are extracted from down-sampled input.

Multi-label segmentation.

One thing not clear: the patch at x plane and y plane share the same weight, which does not make sense to me. Can we assume that CNN must learn same features at orthogonal direction?

### Deep learning for neuroimaging: a validation study
For fMRI etc.

### Classification of Histology Sections via Multispectral Convolutional Sparse Coding
Used convolutional sparse coding + SPM + SVM for image classification. CSC is based on Kavu.. et al. Only one CSC layer, and not deep. If it uses deeper network like U-net, probably do not need SPM and SVM any more, because the features at one pixels should have information of neighboring pixels due to the pooling and unpooling, and 1x1 conv layer can be trained as classifier so no need SVM.

### Hough-CNN: Deep Learning for Segmentation of Deep Brain Regions in MRI and Ultrasound

### Efficient Multi-Scale 3D CNN with Fully Connected CRF for Accurate Brain Lesion Segmentation

### A Fully Convolutional Neural Network for Cardiac Segmentation in Short-Axis MRI

### V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
A 2016 paper on full 3D convnet.



[su15]: http://research.rutgers.edu/~shaoting/paper/MICCAI15_autoencoder.pdf
[bro15]: https://www.cs.ubc.ca/~rtam/Brosch_MICCAI_2015.pdf
