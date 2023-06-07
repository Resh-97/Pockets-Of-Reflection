---
layout: post
title:  Demystifying Hidden Units in Neural Networks through Network DissectionDemystifying Hidden Units in Neural Networks through Network Dissection
date:   2021-05-16
subtitle: Researchers at MIT’s CSAIL elucidate the thought process behind neural network predictions through their fascinating paper <a href=“http://netdissect.csail.mit.edu/”>Network Dissection- Quantifying Interpretability of Deep Visual Representations</a>.
tags: [tag4, tag5]
splash_img_source: /assets/img/UnsplashDemystifying.png
splash_img_caption: Photo by <a href="https://unsplash.com/@alinnnaaaa">Alina Grubnyak</a> on Unsplash.
---
Have you ever wondered how Neural Networks (NN) arrive at predictions once it’s trained? Wouldn’t it be interesting to dissect NN and find out what the hidden units have learned? How do you think hidden units contribute to NN predictions post training? Well, one has plenty of time to think of such intricacies of deep networks when one’s model goes on training. Alas, how can a novice in deep learning put a probe on the hidden units and interpret them. So, I naturally discarded these thoughts until I stumbled upon the paper “[Network Dissection: Quantifying Interpretability of Deep Visual Representations](http://netdissect.csail.mit.edu/final-network-dissection.pdf)”.

About the Paper:
================

Researchers from MIT’s CSAIL propose a technique called “[Network Dissection](http://netdissect.csail.mit.edu/)” where they evaluate every individual convolution unit in CNN on a binary segmentation task to characterize a unit’s behavior. In other words, this method interprets networks by providing meaningful labels to their hidden units. They have shown that interpreted units can be used to provide an explanation for the individual image predictions given by a classifier.

In the past, observations of hidden units have shown that human-interpretable concepts sometimes emerge in individual units within networks. For example, [**object detector units have been observed within scene classification networks**](https://arxiv.org/pdf/1412.6856.pdf) and [**part detectors have emerged in visual recognition tasks**](https://arxiv.org/pdf/1607.03738.pdf)**.**

Using  ‘Network Dissection’, the authors evaluate the emergence of such concept detectors in deep networks, quantify the interpretability of individual units in CNNs and attempt to answer the question - ‘_Do CNNs learn disentangled features? ’.
Note: Disentangled features are narrowly defined hidden units that encode specific real world concepts._

Network Dissection Method:
==========================

The interpretability of individual units is quantified by measuring the alignment between a hidden unit’s response and a set of visual concepts. Human-interpretable concepts include low-level concepts like colors and high-level concepts such as objects. By measuring the concept that best matches each unit, Net Dissection can break down the types of concepts represented in a layer.

Quantifying interpretability for individual units using Network Dissection proceeds in three steps:

**1: Gather images with human-labeled visual concepts.**
--------------------------------------------------------

To identify ground truth exemplars for a broad set of visual concepts, the authors assembled a new heterogeneous dataset called **_Broden_.**

> **_The Broadly and Densely Labeled Dataset_** _(_**_Broden_**_)_ unifies several densely labeled image data sets: [ADE](https://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf) , [Open Surfaces](https://www.cs.cornell.edu/~sbell/pdf/siggraph2014-intrinsic.pdf) , [Pascal-Context](https://www.cs.toronto.edu/~urtasun/publications/mottaghi_et_al_cvpr14.pdf) , [Pascal-Part](https://arxiv.org/pdf/1406.2031.pdf) and [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/publications/2014/Cimpoi14/cimpoi14.pdf). These data sets contain examples of a broad range of objects, scenes, object parts, textures, and materials in a variety of contexts.

Figure: A sample of the types of labels in the Broden dataset. ( Figure from [Bau & Zhou et. al (2017)](http://netdissect.csail.mit.edu/final-network-dissection.pdf) )

There are around 60,00 images in the Broden dataset and annotations spanning 1197 visual concepts. Images are pixel-wise labelled for most visual concepts, except texture and scene where individual labels are given for the full image. Additionally, every image pixel is labelled with one of the 11 common color names. This way every image will get an annotation mask, **L**\_c for every visual concept, **c**.

2: Retrieve individual units’ activations**.**
----------------------------------------------

To gather the response of individual units to concepts, images from the Broden dataset are fed into the CNN and a forward pass is performed.

1.  For the each convolutional unit (**k**), feed each input image (**x**) from the Broden dataset to the CNN and compute the activation map, **A\_**k(**x**).
    _Activation maps are the outputs of the unit post a convolution operation.
    _**_Note:_** _In a unit, a kernel or filter is convolved with the image volume._
2.  Calculate the distribution of activation, **a\_**k, over all images. **a\_**k is a real valued map.
3.  To convert it into a binary map, compute a top quantile threshold **T\_**k , such that P(**a\_**k >**T\_**k)=0.005. This means 0.5% of all activations of unit ‘**k**’ for image **x** is greater than **T\_**k.
4.  Generally, deeper into the NN , smaller the size of the activation map. In order to obtain a binary segmentation map, use bilinear interpolation to scale the lower-resolution activation maps, **A**\_k(**x**) to the image resolution resulting in **S**\_k(**x**).
5.  Binarize the activation map: A new mask, **M**\_k(**x**)=**S**\_k(**x**)≥**T**\_k(**x**), is obtained such that a pixel is on or off depending on whether it exceeds the activation threshold **T**\_k.
    _Note: These activation masks mark the highly activated areas._

**3: Quantify activation−concept alignment.**
---------------------------------------------

Now we have human-labelled concept mask, **L**\_c (from step 1) and activation mask, **M**\_k (from step 2). Next, we need to identify the visual concept that activates a particular node. In other words, we try to identify which concept each node is “looking for”.

This is done by comparing the activation masks with all labeled concepts. We quantify the alignment between activation mask, **M**\_k and concept mask, **L**\_c with the **Intersection over Union (IoU)** score.

Figure: **Intersection over Union (IoU)** score formula.

**IoU score** =( Number of pixels identified by both the masks as concept **c** ) **/**
( Total number of unique pixels identified as concept **c**)

Figure: Example of how **IoU Score** is computed. (Source: [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/cnn-features.html#network-dissection))

> The value of **IoU**\_(k,c) is the accuracy of unit **k** in detecting concept **c.** We consider **k** as a detector of concept **c** if the **IoU score exceeds a threshold.**

The authors chose 0.04 as the threshold to classify a unit as a particular concept detector. One unit could detect multiple concepts and for analysis the top ranked label was chosen.

To quantify the interpretability of a layer, the number unique concepts identified in the layer was noted as **number of unique detectors**.

Experiments:
============

With the framework set out, the authors tested Net Dissection on different network architectures (AlexNet, GoogLeNet, VGG, ResNet) trained from scratch on different datasets (ImageNet, Places205, Places365). For self-supervised training tasks, AlexNet was trained for tasks such as solving puzzles and tracking.

**_ImageNet_**  is an _object-centric_ dataset with 1.2 million images from 1000 classes.  **_Places205_** and **_Places365_** is a _scene-centric_ dataset with 205 and 365 categories respectively. Places205 contains 2.4 million images and Places365 contains 1.6 million images from categories like kitchen and living room.

Following are some of their findings:
-------------------------------------

1.  The authors found detectors of high-level concepts at higher layers and low-level concepts at lower layers (i.e. low-level concepts like _color_ and _texture_ dominated at _conv1_ and _conv2,_ while more _object_ and _part_ detectors emerged in _conv5)._
2.  Networks trained on supervised tasks have more unique detectors than those trained on self-supervised tasks.
3.  The number of unique concept detectors increases with the number of training iterations.
4.  Batch normalization reduces the number of unique concept detectors while increasing the number of units in a layer increases the number of interpretable units.
5.  Interpretability of ResNet > VGG > GoogLeNet > AlexNet. Interpretability of models trained on Places > ImageNet.

Figure from [Bau & Zhou, et al. 2017](http://netdissect.csail.mit.edu/final-network-dissection.pdf).

Conclusion:
===========

Network Dissection helps us understand what emergent concepts appear in a NN, allowing us to quantify its interpretability. Though concept detectors emerged within the network, not all the units in a NN were interpretable which proved a partial disentangled representation within the network.

Figure: Total Number of interpretable units in a layer. ( Figure from [Bau & Zhou et. al (2017)](http://netdissect.csail.mit.edu/final-network-dissection.pdf) )

> The authors also used Network Dissection for Generative Adversarial Networks (GANs). You can find the project [here](https://gandissect.csail.mit.edu/).

Hope you found this article helpful. Thankyou for Reading!
