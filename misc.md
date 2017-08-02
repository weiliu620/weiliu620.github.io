

### Semantic Component Analysis
Each component is constrained to have semantic meanings. See videolecture.net. Can be used for multi-channel image decomposition/segmentation. (energy dispersion imaging)

### Training Deep Networks with Structured Layers by Matrix Backpropagation
This paper cited the bilinear CNN paper. Add a structure matrix layer (SVD) after CNN layers, and have a lot of theoretical proof. Worth reading?

### Receptive fields of single neurones in the cat’s striate cortex
This is the original Hubel and Wiesel's work. May not read it but need to know this paper anyway.

### Fiber Connection Pattern­ guided Struc­tured Sparse Representation of Whole-brain FMRI Signals for Functional Network Inference
Not really a deep learning paper but...

### Nonlinear Regression on Riemannian Manifolds and Its Applications to Neuro-Image Analysis
May note be a deep learning paper. Add it anyway.

### Predicting Activation across Individuals with Resting-State Functional Connectivity based Multi-Atlas Label Fusion

### Emergence of simple-cell receptive field properties by learning a sparse code for natural images
An early paper about human vision that many CNN works are based on.

### Oriented Edge Forests for Boundary Detection

### Multimodal Cortical Parcellation Based on Anatomical and Functional Brain Connectivity

### Multilevel Parcellation of the Cerebral Cortex Using Resting-State fMRI

### Fast High-Dimensional Filtering Using the Permutohedral Lattice
Read this paper because of another one "Conditional Random Fields as Recurrent Neural Networks". This paper proposes to to the filtering in the feature space. Each pixel has a feature vector, and all the pixels can be transformed into the high dimensional feature space. The data points are blurred and re-sampled, then mapped back to image space. Bilateral filters can be seen as a special case of the this definition, because the weights with which neighboring pixels contribute to the averaging depends on both the spatial distance and the pixel intensity between the center pixel and neighbor pixel. When the  spatial distance and pixel intensity difference are obtained from another image, bilaterial filtering can be used to filtering image A without crossing the edge of image B.

That remind me in fMRI, we can apply spatial smoothing filter on functional image but use bilateral filter with the weights defined by structure image such as T1 or T2. This is better than simple Gaussian smoothing, because the smoothing will not cross boundary between gray and white matter of structure image. Make sense?

### Reconstructing visual experiences from brain activity evoked by natural movies
Cited from "Inverting Visual Representations with Convolutional Networks.pdf". Seems the overlapped area of ML and neuroscience.
