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

- **Conv2D -> BatchNorm -> ReLU.** BatchNorm normalizes pre-activation values before the nonlinearity.
- **One residual connection.** Deep networks can lose gradient signal on the way back to early layers (vanishing gradients), stalling their learning. A skip connection gives the gradient a direct path around two conv layers. One block is enough to see the idea at this scale; a full ResNet stack would be excessive for a 10-class, 32x32 problem.
- **GlobalAveragePooling2D** Averages each feature map to one number - this is what keeps the parameter count low and pushes the conv layers to learn what matters, rather than leaving that to a huge Dense layer.
- **Learning rate 0.0001, not Keras's 0.001 default.** Carried over from a debugging finding on the DNN and confirmed to hold here too.

**Parameters: 169,802** - about a fifth of the DNN's, in a deeper architecture.

## 5. Setup and Usage
 
```bash
pip install -r requirements.txt
```
 
```python
from data_util import load_image_data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_image_data()
```
 
Downloads SVHN via `tensorflow_datasets` on the first run, and caches locally after. Run either notebook till the end.

## 6. Results
 
| | DNN | CNN |
|---|---:|---:|
| Test accuracy | 70.6% | **90.1%** |
| Parameters | 820,874 | 169,802 |
 
 
### Where the CNN's gains actually came from
 
| Digit | DNN recall | CNN recall | Gain |
|---|---:|---:|---:|
| 0 | 0.60 | 0.90 | +0.30 |
| 1 | 0.88 | 0.96 | +0.08 |
| 2 | 0.82 | 0.94 | +0.12 |
| 3 | 0.60 | 0.86 | +0.26 |
| 4 | 0.80 | 0.92 | +0.12 |
| 5 | 0.54 | 0.87 | +0.33 |
| 6 | 0.60 | 0.86 | +0.26 |
| 7 | 0.66 | 0.92 | +0.26 |
| 8 | 0.57 | 0.78 | +0.21 |
| 9 | 0.59 | 0.85 | +0.26 |

The biggest gains for the CNN are for the digits that the DNN struggled with the most (for example, 5,3,6,7, and 9).

### A debugging note
 
The DNN's first run stalled flat at ~19% accuracy, with every prediction collapsing to the most common digit in the dataset. While this looked like a data bug at first - it was ruled out by checking the label distribution and spot-checking images against labels directly. The real cause was that Keras's default learning rate (0.001) was too large for a network with a very wide first Dense layer (3,072 flattened inputs into 256 neurons), so training oscillated instead of converging. Dropping to a learning rate of 0.0001 fixed it immediately. The same value was used with the CNN as a starting point and confirmed it worked there too.
 
## 7. Known Limitations
 
- No transfer learning comparison.
- One residual block, not a full ResNet stack.
- No hyperparameter tuning.

## License
MIT