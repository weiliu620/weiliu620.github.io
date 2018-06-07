---
title: Recognition related topics
---
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
### Pyramid Match Kernels: Discriminative Classification with Sets of Image Features

This is an early work of object recognition, before deep learning age. At that time, the steps of object recognition or scene classification usually take the following steps:

- Feature detection. Can be detectors such as Harris or SIFT, or sampled from a uniformly sampled grid.

- Build feature descriptor at each detected feature location. Can be SIFT descriptor or simple oriented edges.

- Calculate the similarity between two images with the feature descriptors. The similarity can be defined by the correspondence of two sets of features. However, the cardinality of the sets can be different, and finding the (partial) correspondence is a hard problem.

- Use the similarity as a kernel matrix of the SVM classifier and train the classifier.

The pyramid matching kernels are proposed to calculate the partial matchings of two sets of feature descriptors efficiently.

- First, define a $$d$$-dimensional histogram in the $$d$$-dimensional feature space. The max value of feature at each dimension is normalized to be $$D$$. Then define a pyramid of histograms, lower resolution histogram double the bin size from higher resolution histogram.
- Then define a histogram intersection function as the min of the number of features that fall into same bin. We take the min not the max in order to build a Mercer kernel.
- The kernel function is define as the weighted sum of the histogram intersection at all levels. because coarser level intersections include all intersections at finer level, only the new matched features are counted, i.e. weighted sum of the difference between previous level and next level.

Here are the properties of the pyramid matching kernel:
- The correspondence based matching is optimal in the sense that the overall similarity is maximized. The proposed pyramid kernel is an approximation of this optimal matching.
- It's efficient.

On first look, when $$d$$ is big, such a high dimensional histogram seems difficult to compute. It turns out computing the histogram and the kernel function is very efficient.

The experiment of using ETH-80 mentioned that this is a categorization task, not an object recognition task. It looks this dataset has 8 classes, but has 10 objects. So some class may have more than one object.

  My understanding is if an instance of an object exists in training set and we try to find same object in test set, this will be a object recognition task. If training set has a some object in a class, but test set has a different object in the same class, and we try to decide which class the new object is in, this will be a categorization task.

### Beyond bags of features/ Spatial pyramid matching for recognizing natural scene categories

This paper use pyramid matching function to calculate the matchings between feature sets of two images. The difference with previous pyramid matching kernel method is that here the pyramid is built in the image space, not feature space.

- build dense SIFT features on training set, apply K-Means on a random subset of patches from training set, and learn a visual vocabulary (with size $$M$$)
- Given a test image, define the interest points on uniformly sampled coordinates on the grid. Calculate SIFT features on the and get the feature descriptors.
- Project the feature descriptors onto the dictionary, and get the label of closet cluster center to represent that feature descriptor.
- Use the cluster label as new features, and apply the pyramid matching on the spatial domain. So the histogram is in lower (2) dimension.
- Use the matching kernel function as the similarity, and apply SVM for classification.

### Aggregating local descriptors into a compact image representation (VLAD)
This paper propose an efficient method to query a test image and find the best matched images in the database. To do that, we need to represent samples in database with a feature vector, then reduce the dimension of the feature vector, then *index* the feature vector with a binary vector.

*Bag of Features/Visual words*: First build a codebook with methods like K-Means on training set. Given a test image, extract the feature descriptor and project it onto the centroid of the clusters, and get the class assignment label. Build the histogram of this class assignment labels with all feature descriptors, and use this histogram as the final feature of current image.

*Fisher vector*: First train a parametric model (such as GMM) from training set. The feature descriptor is defined as the the gradient of sample's likelihood with respect to the parameters.

The VLAD is a simplification of Fisher vector. Again, we learn a codebook from training data. Then for each feature descriptor defined on the input image, we calculate the residual or the distance of the descriptor from the closest code, and sum over all feature descriptors for each cluster. Then we concatenate all $$K$$ feature descriptor sums. So if originally using SIFT feature of length 128, the total VLAD feature vector will be $$K\times 128$$ dimension.

The next step is to convert the feature vector to a binary code. The paper claims that the product quantization based search method is better than locality sensitivity hashing method. There are two possible errors: one is introduced during dimension reduction, and the second one is introduced during quantization. For quantization, the input feature vector is not quantized, but the image in database are quantized by a $$q$$ function. The squared loss function is decomposed by the sum over the squared loss for each smaller component between input $$x$$ and database image $$q(y)$$. (One thing is not clear, why input $$x$$ is not quantized? $$x$$ is real numbers while $$q(y)$$ is binary).

The paper also talked about optimal choice of the target dimension of PCA, to achieve the trade-off between two errors.

Overall, it seems the VLAD descriptor is the main contribution of this paper.

### Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition

The paper pointed out the steps of standard steps of image classification and object detection tasks:

- Extract features. Traditional methods may find interest points, and only extract features on those sparse points. There are dense SIFT that probably extract feature vectors at each pixel. Modern convnet can also been seen as a feature extraction step if we use the output of the last convolution layer.

- Encoding, such as vector quantization, sparse coding, Fisher kernels.

- Merging. Can be bag of words, or spatial pyramid pooling.

Pyramid matching kernel defines the histogram in feature space, spatial pyramid pooling define the histogram on image space. Also, PMK use the original feature vectors, while SPM uses the quantized features (a set of (x, y) points for each type of feature).

SPP uses original features without quantization. In this sense, it is similar to PMK. But SPP define the histogram on image domain (hence the 'spatial' in SPP), and in this sense is similar to SPM.

If the size of the codebook is $$M$$ (as defined in SPM paper), then the length of the pooled feature vector is also of length $$M$$ (I believe). The paper also pointed out that at the highest level, the spatial histogram only has one bin. For each feature, we just average the feature's value (a real number) across spatial domain. And this is equivalent to the 'global average pooling'.

The actual implementation is usually done similar to max-pooling layer. 
