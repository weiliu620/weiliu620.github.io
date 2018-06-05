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

### Some feature representation methods

Bag of Features/Visual words

Fisher Vector:
