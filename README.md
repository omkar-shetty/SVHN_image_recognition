# SVHN Digit Classification: DNN vs. CNN

This repo includes a comparison of two architectures on the same digit-classification problem, built to understand the imapct of architecture choice for image data. The same data, same splits, and same evaluation code are used for both models with architecture being the only major difference between the two.

## 1. Problem Statement

[SVHN](http://ufldl.stanford.edu/housenumbers/) is a 10-class digit recognition dataset built from real photos of house numbers, cropped to individual digits. The usgae of real images makes this dataset harder than MNIST due to the color images, cluttered backgrounds, and inconsistent lighting. In addition, the data is skewed with more instances of the digits 1 and 2 compared to the others.
 
- **Classes:** 10 (digits 0-9)
- **Train:** 65,932 images
- **Validation:** 7,325 images
- **Test:** 26,032 images
- **Image shape:** 32x32x3
Both models load identical splits via a single shared `load_image_data()` function.

## 2. Significance of a CNN vs. DNN

In tasks like digit recognition, the spatial relationship between pixels is critical - for example, the edges and curves that help identify a "3" from an "8".

However, for a fully connected network, flattening the image into a vector leads to the model losing this important information.

In contrast, a CNN's convolutional layers slide a small filter across local neighborhoods, and reuse the same filter's weights at every position. That preserves spatial relationships (and with far fewer parameters), since one filter is shared across the whole image rather than needing a separate weight per pixel-to-neuron connection.

## 3. Architecture - DNN Baseline
 
```
Input (32, 32, 3)
  -> Flatten
  -> Dense(256, relu) -> Dropout(0.3)
  -> Dense(128, relu)
  -> Dense(10, softmax)
```
 
Sequential API - a genuinely linear stack, nothing to branch, no reason to reach for more.
 
**Parameters: 820,874**

## 4. Architecture - CNN
 
```
Input (32, 32, 3)
  -> Conv2D(32) -> BatchNorm -> ReLU -> MaxPool
  -> Conv2D(64) -> BatchNorm -> ReLU -> MaxPool
  -> Residual Block(64): Conv-BN-ReLU-Conv-BN -> Add(shortcut) -> ReLU
  -> Conv2D(128) -> BatchNorm -> ReLU -> MaxPool
  -> GlobalAveragePooling2D -> Dropout(0.3)
  -> Dense(10, softmax)
```
In contrast to the DNN, here a functional API is used - primarily to support the residual block's `Add()` step which needs its own input later in the graph.