---
title: general reading notes
---
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


### Semantic Component Analysis
Each component is constrained to have semantic meanings. See videolecture.net. Can be used for multi-channel image decomposition/segmentation. (energy dispersion imaging)

### Training Deep Networks with Structured Layers by Matrix Backpropagation
This paper cited the bilinear CNN paper. Add a structure matrix layer (SVD) after CNN layers, and have a lot of theoretical proof. Worth reading?

### Receptive fields of single neurons in the cat’s striate cortex
This is the original Hubel and Wiesel's work. May not read it but need to know this paper anyway.

### Fiber Connection Pattern­ guided Struc­tured Sparse Representation of Whole-brain FMRI Signals for Functional Network Inference
Not really a deep learning paper but my past research on fMRI.

### Nonlinear Regression on Riemannian Manifolds and Its Applications to Neuro-Image Analysis
May not be a deep learning paper. Add it anyway.

### Predicting Activation across Individuals with Resting-State Functional Connectivity based Multi-Atlas Label Fusion

### Emergence of simple-cell receptive field properties by learning a sparse code for natural images
An early paper about human vision that many CNN works are based on.

### Oriented Edge Forests for Boundary Detection

### Multimodal Cortical Parcellation Based on Anatomical and Functional Brain Connectivity

### Multilevel Parcellation of the Cerebral Cortex Using Resting-State fMRI

### Fast High-Dimensional Filtering Using the Permutohedral Lattice
Read this paper because of another one "Conditional Random Fields as Recurrent Neural Networks". This paper proposes to to the filtering in the feature space. Each pixel has a feature vector, and all the pixels can be transformed into the high dimensional feature space. The data points are blurred and re-sampled, then mapped back to image space. Bilateral filters can be seen as a special case of the this definition, because the weights with which neighboring pixels contribute to the averaging depends on both the spatial distance and the pixel intensity between the center pixel and neighbor pixel. When the  spatial distance and pixel intensity difference are obtained from another image, bilateral filtering can be used to filtering image A without crossing the edge of image B.

That remind me in fMRI, we can apply spatial smoothing filter on functional image but use bilateral filter with the weights defined by structure image such as T1 or T2. This is better than simple Gaussian smoothing, because the smoothing will not cross boundary between gray and white matter of structure image. Make sense?

### Reconstructing visual experiences from brain activity evoked by natural movies
Cited from "Inverting Visual Representations with Convolutional Networks.pdf". Seems the overlapped area of ML and neuroscience.


### [Generative Adversarial Nets](http://arxiv.org/abs/1406.2661)
This paper proposed a model to generate new examples given existing data. The model includes a generative model $$G$$ and a discriminative model $$D$$. The model $$G(z)$$ generates new example $$x$$ from random input $$z$$, and model $$D(x)$$ output the probability of the example $$x$$ being from the true data distribution.

The training includes an inner loop and outer loop. In the inner loop, model $$G$$ is fixed, and $$D$$ is optimized to correctly label both the true observed examples and those generated form $$G$$. Once the inner loop converges, $$D$$ is kept fixed. Then in the outer loop, $$G$$ is optimized to minimize the probability that $$D$$ correctly labels the generated examples. When this *minmax* algorithm converges, the distribution of generated sample $$x$$ from $$G$$, is equal to the true data distribution. When this happens, $$D$$ equals to $$1/2$$ for any example $$x$$.

In practice, both model $$G$$ and $$D$$ are multilayer perceptrons, and the optimization of both models are not on the entire space of possible models, but on the parameters of MLP.

### [An analysis of single-layer networks in unsupervised feature learning](http://ai.stanford.edu/~ang/papers/nipsdlufl10-AnalysisSingleLayerUnsupervisedFeatureLearning.pdf)

This is an earlier work of the [paper below][1]. The paper uses different methods for feature extraction, including sparse autoencoder, sparse RBM, K-means and Gaussian mixture model. After the feature mapping is learned, a single-layer model map the input data into feature space and does classification.

After the feature mapping, we get feature map of size $$M \times M \times K$$, where $$M$$ is the size of the single feature map and $$K$$ is number of filters. Then each $$M \times M$$ feature map is split into four quadrants, and compute the sum of each quadrant as a pooling. So the final feature map is $$4K$$ which 4 numbers at each feature map. Such pooling method is different from the more recent pooling.

The main goal of the paper, is to see the effect of some hyperparameters on the classification results. These parameters are number of filters $$K$$, the stride $$s$$, receptive filed size (that is, the filter size) $$w$$ and also the whitening as a preprocessing step. The experiments seem to show that these parameters have larger effect on the classification results compared to the choice of feature learning methods. In particular, when using K-means as feature extraction methods, as long as the hyperparameters are chosen correctly, the algorithm get the state-of-the-art results. (This may not be true in year 2016, I think.)

### [The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization][1]

This paper defines feature learning in two steps: 1) Learn a set of basis functions as dictionaries $$D$$. 2) An encoding algorithm that learns a mapping $$f$$ from the data $$x$$ to the feature space, given the dictionary $$D$$. For example, the first step can use simple methods like K-means to learn the dictionary, or more advanced methods like sparse coding, or sparse autoencoders. The second step also can be sparse coding, orthogonal matching pursuit, etc.

The paper choose the combinations of methods in step 1 and those in step 2, and compare the performance. The results in this paper and previous work show that the choice of dictionaries are not critical. For example, even random chosen orthogonal vectors as dictionary can get good results.

The paper also shows that sparse coding used for both steps achieves better results than other combination of methods when there is not many labeled data.

Question: how the labels are used for training in both steps? First step should be unsupervised, so no labeled data is required. So labels are used in encoding step? That does not make much sense.

### [Convolutional Clustering for Unsupervised Learning]()
This paper have two main contributions on the feature learning and the convnet model structures, respectively. First, the feature learning is based on K-means. When the conventional K-means method is used for learning the dictionaries, a collection of patches is randomly selected from the training data, and the patch is the same dimension as the dictionaries. In this work, a random location is selected first from a random selected input image, a patch of twice the size of the dictionary dimension is extracted around this location. The current filters are applied on this patch in a Convolutional way. the new location within the patch is selected where the activation is largest, and a new patch is defined and extracted at this new location, so the new patch has the same size as the filter. The author claims that such method can remove the redundancy within the dictionaries.

the second part of the paper is to learn the sparse connection between features of one layer to those of next layer. It looks the majority of convnet now days just use full connected layers. The authors of this paper claim such full connected layer is not optimal. To learn the connection, the authors define a connection matrix which maps the previous feature to next layer. (details not read)

### [Understanding Deep Image Representations by Inverting Them (Nov 2014)][mah14]

- Inversion method works for SIFT, HOG and also convnet. Inversion is defined as a regularized regression problem.
- Implement SIFT and HOG as a special type of convnet.
- The main purpose of this paper is to see how much information remains in the feature maps of each layer. The purpose is not to construct an image that best match the target image. Otherwise, we can simply use the target image as initial input, and the algorithm will converge at the first iteration, because the cost function already at its lowest value.

On the algorithm side, define a feature transform (learned from training data) $$\phi(x)$$ on input image $$x$$. Given the transform $$\phi(x_0)$$ for target image $$x_0$$, the cost function is defined as the norm of the difference between $$\phi(x_0)$$ and the feature map $$\phi(x)$$ of input image $$x$$,

$$
L(x) = || \phi(x) - \phi(x_0)||^2_2 + R(x)
$$

### Intriguing properties of neural networks
Adding noise to image will change the classification results, even the noise is imperceptible to human vision. However, the injected noise is high-pass structured noise which can rarely occur in natural images. (found from [mah14])

### Visualizing and understanding convolutional networks
backtrack the network computations to identify which image patches are responsible for certain neural activations. Compared to [dos15], this paper also uses additional information: the max location of the intermediate layers.

### [Inverting Visual Representations with Convolutional Networks][dos15] (CVPR 2016)
- This paper also generates images from a trained convnet. Instead of finding an image $$x$$ that minimize the Euclidean distance between $$\phi(x)$$ and $$\phi(x_0)$$, as in [mah14],  this paper explicitly minimize the distance between $$x$$ and $$x_0$$, where $$x_0$$ is the target image.
- [mah14] need optimization at test, but this work does not. It only need forward pass during test.
- Define a Up-convnet $$f(\phi, W)$$ for inversion, where $$\phi$$ is the feature map, and $$W$$ is parameter of $$f$$. Optimize $$W$$ to minimize
$$ || x_i - f(\Phi(x_i))||^2_2 $$.
- Also implement HOG, SIFT and other traditional feature representations with convnet.

### [Reconstructing visual experiences from brain activity evoked by natural movies][nis11]
Found this work from [dos15] which potentially can be used for inverting signal from real brain, like in this paper.

### Learning to Generate Chairs, Tables and Cars with Convolutional Networks
- Based on the knowledge learned from other chairs, the model is able to generate a chair with new view angle, given an example of this chair in one view angle.
- The train sample is opposite to the normal training samples: input is $$c, v, \theta$$, where $$c$$ is the class label, $$v$$ is the view angle, and $$\theta$$ is other transform parameters, such as scaling, shifting, rotation, etc. Output is $$(x, s)$$, where $$x$$ is the full image and $$s$$ is the mask of the object in the image.
- The model definition: The three input vector are fed into a few full connection layers then concated into a long feature vector. The feature enters an **unconvolutional network**, which is indeed a upsampling (unpooling) step. To learn the parameters of this network, the cost fucntion is defined as the error of reconstruction.

This paper is by the same author of [dos15], but I could not find how they referenced each other...

### Spatial Transformer Networks
Explicitly define a special layer for the spatial transformation such as rotation, scaling, etc. The parameter of this layer is learned together with the rest of the network.

Possible application: This can potentially be used on medical image segmentation. In medical imaging, segmentation and registration are coupled problems. An ideal model can optimize the segmentation and registration in a single cost function. A fully convolutional net can be used for segmentation. Together with this spatial transformation layer, we can probably have a integrated model for both segmentation and registration.

### Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
Really should read this paper.

### Learning to Transduce with Unbounded Memory
Talks about neural stacks. Need to take a look.

### Deep Neural Decision Forests
Found this talk on ICCV 2015 videos. Use random forests for classifier and learn it with CNN.

### 3D Convolutional Neural Networks for Human Action Recognition
An early paper of 3D convnet. Need to check who cited it, and how the 3D conv works.

### Inceptionism: Going deeper into neural networks, 2015

### Rethinking the Inception Architecture for Computer Vision (late 2015)
Proposed use 1x3 and 3x1 filter to replace 3x3 conv filter. Can be used to approximate 3D filter, too? Is that smae thing with separable filter. Need to read details.

### Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (Feb 2016)

### Batch normalization: Accelerating deep network training by reducing internal covariate shift

### Rich feature hierarchies for accurate object detection and semantic segmentation

### Predicting Eye Fixations using Convolutional Neural Networks
About visual attention and saliency.

### Explaining and harnessing adversarial examples

### How transferable are features in deep neural networks
Talks about which layer to transfer in what situation beween orignal task and new task.

### Deep neural networks are easily fooled: High confidence predictions for unrecognizable images

### From generic to specific deep representations for visual recognition

### Learning transferable features with deep adaptation networks
have a RKHS to embed the feature in source domain and help the task in target domain.

### Unsupervised Learning of Invariant Feature Hierarchies with Applications to Object Recognition (2007)
An early paper for unsupervised learning using autoencoder. But also introduced a tranformation parameter $$U$$ that can be optimized. So the encoder has parameter $$W_C$$ and $$U$$, and decoder has parameter $$W_D$$ and $$U$$. Also introduced a hidden/auxiliary variable and a cost term of the norm between the encoding and the hidden variable, and use EM-like optimization. The total cost function is like $$ || Y - Dec(Z, U; W_D) || + \alpha || Z - Enc(Y, U; W_C) || $$.

The hyperparameter $$\alpha$$ is set to one in the paper. I'm thinking if $$\alpha = 0$$, the model does not have any constraints on the form of $$Z$$, isn't it that we just get a dictionary learning (without regularization yet)? In that case, we cannot learn $$W_C$$ either. So $$\alpha$$ have a continuous control between a autoencoder model, and a dictionary learning (or, sparse coding) model. Is that right?

Compared to sparse coding, this autoencoder does not need optimization step during test, because of the encoder available for a simple forward pass.

The additional transformation parameter $$U$$ is reminiscent of the more recent spatial transormer network, which not only model shifting but also rotation and other thansforms. Also, the $$U$$ in this paper is not learned, but just copied from the encoding step to the decoding.

Also, because of the introduction of $$U$$, no two filters are shifted version of each other. (future convolutional autoencoder can achieve this, too)

During training, the patches are randomly selected for training images. Also, it seems some parameters of the sparsity component are constrained during the training, but relaxed during test, because the constrained form does not have enough information for classification.

The autoencoder includes two hidden encoding layers. the second layer selectively connects to the feature map of second layer. This may be because of the computational cost, though the autho didn't say the reason.

### Sparse Feature Learning for Deep Belief Networks (2008)
Have read it but don't remember details. Basically it's encoder-decoder model using RBM-like machines. Learning is still EM.

### Fisher Vectors Meet Neural Networks: A Hybrid Classification Architecture

### Deformable Part Models are Convolutional Neural Networks (2015)
CVPR paper that I haven't got chance to read.

### DeepContour: A Deep Convolutional Feature Learned by Positive-sharing Loss for Contour Detection

### DeepEdge: A Multi-Scale Bifurcated Deep Network for Top-Down Contour Detection

### Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks (2014)
Transfer learning.

### CNN Features Off-the-Shelf: An Astounding Baseline for Recognition (2014)
Use pretrained net as a feature extractor and train linear SVM on top.

### From generic to specific deep representations for visual recognition (2014)

#### XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks
Latest binary network. Can run on cpu.

### Deep Networks with Stochastic Depth

### Generative Image Modeling using Style and Structure Adversarial Networks

### Deep Learning in Bioinformatics

### Towards Bayesian Deep Learning: A Survey

### Structured and Efficient Variational Deep Learning with Matrix Gaussian Posteriors

### Expectation Backpropagation: Parameter-Free Training of Multilayer Neural Networks with Continuous or Discrete Weights

### Deconvolutional Networks
An old paper. The name of the paper is a bit confusing: although it is named as deconvolutional networks, it is really a convolutinoal sparse coding. The input image is assumed to be the sum of product between a hidden feature map and a filter. The hidden feature map is larger than the input image due to the convolution. The author calls such convolutional sparse coding as top-down, and call the conventional LeCunn convolution as bottom-up.

### Fast, Exact and Multi-Scale Inference for Semantic Image Segmentation with Deep Gaussian CRFs
Solve Gaussian MRF by linear equation. Implemented the solver on GPU, and also use multi-scale. Seems too complicated.

### Evaluating the visualization of what a Deep Neural Network has learned
Mentioned that "Layer-wise Relevance Propagation" is a better method for visualization of CNN, than the old deconvolution method.

After a quick look, not that interesting to me. The work is to find which set of pixels are most important for classificaiton.

### Any VLAD paper

### Going deeper with convolutions
GoogleNet papers. should I read it?

### Intriguing properties of neural networks
An early papers that shows a small pertubation to input image can have big impact on output of deep network.

### Deep neural networks are easily fooled: High confidence predictions for unrecognizable images

### Deep Residual Learning for Image Recognition

### Identity Mappings in Deep Residual Networks
A follow up of the residual net.

### Highway networks

### UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL
GENERATIVE ADVERSARIAL NETWORKS

### Discriminative Regularization for Generative Models

### Feedforward semantic segmentation with zoom-out features

### Composing graphical models with neural networks for structured representations and fast inference

### Deep Convolutional Inverse Graphics Network

### Adversarial Autoencoders
The discriminator take the hidden variable z as input (either from the encoder of AE, as negative sample, or from a prior distribution, as positive example). Replaced the KL divergence cost function with the Adversarial. This is because, according to the paper, that original KL optimize $$q(z)$$ to pick the modes of $$p(z)$$, but Adversarial will optimize $$q(z)$$ to the whole distribution of $$p(z)$$.

LW: if that is the true reason of using Adversarial, we can also think about expectation Propagation, which seems to match q(z) to the whole P(z), too.

The optimization is iterative. In both the Adversarial optimization step and the the AE step, the encoder is optimized. In Adversarial step, encoder is optimized to match q(z) to p(z). In AE step, encoder is optimized to minimize the reconstruction error. In other Adversarial+AE work, there is often a trade off between the two goal of optimization, but here the two optimization step are done iteratively. How to achieve the trade-off for the encoder?

### Adversarially Learned Inference
Given data, generate hidden representations; Given hidden z, generate data. Use discriminator net to tell the two pairs apart.

### Composing graphical models with neural networks for structured representations and fast inference

### Improved Techniques for Training GANs

### Photo Stylistic Brush: Robust Style Transfer via Superpixel-Based Bipartite Graph


[1]: http://people.ee.duke.edu/~lcarin/icml11-EncodingVsTraining.pdf
[mah14]: http://arxiv.org/abs/1412.0035
[dos15]: http://arxiv.org/abs/1506.02753
[nis11]: http://www.cell.com/current-biology/abstract/S0960-9822(11)00937-7
