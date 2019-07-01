# DCCA: Deep Canonical Correlation Analysis

Python implementation of Deep Canonical Correlation Analysis (DCCA or Deep CCA).
The model is implemented using the Keras functional API. The loss function is implemented in tensorflow.

This implementation is based on the availale [Theano DCCA](https://github.com/VahidooX/DeepCCA) and [Tensorflow DCCA](https://github.com/adrianna1211/DeepCCA_tensorflow) implementations.


**Abstract:** "Deep Canonical Correlation Analysis (DCCA) is a method that learns complex non-linear transformations of two views of data such that the resulting representations are highly linearly correlated. Parameters of both transformations are jointly learned to maximize the (regularized) total correlation. It can be viewed as a non-linear extension of the linear method canonical correlation analysis (CCA). It is an alternative to the nonparametric method kernel canonical correlation analysis (KCCA) for learning correlated nonlinear transformations. Unlike KCCA, DCCA does not require an inner product, and has the advantages of a parametric method: training time scales well with data size and the training data need not be referenced when computing the representations of unseen instances.

Galen Andrew, Raman Arora, Jeff Bilmes, Karen Livescu, "[Deep Canonical Correlation Analysis.](http://www.jmlr.org/proceedings/papers/v28/andrew13.pdf)", ICML, 2013.
