# **Traffic Sign Recognition**

---

**Building a Convolutional Neural Net to Correctly Classify Traffic Signs**

The goal of this project is to create a convolutional neural network capable of classifying a set of traffic signs, originally taken from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). From the website, this set is characterized as a single-image, multi-class classification problem, composed of more than 50,000 unique images of traffic signs commonly found in Germany. Before attempting to design the solution, however, I am going to spend some time getting to know the data I'm working with.

## Dataset Exploration

The dataset I'll be using is broken into the following groups:
* 34,799 Training examples
* 4,410 Validation examples
* 12,630 Testing examples

The images are RGB, 32x32 pixels in dimension. There are a total of 43 separate classes represented. Using Matplotlib's `hist` method, we can visualize how evenly the various classes are distributed.

<img src="./examples/histogram.png" />

Here we see that some images are represented far more frequently than others. The class labeled 0 ("Speed limit (20km/h)") in the dataset is the least represented, with only 210 examples.

<img src="./examples/class-0-examples.png" />

On the other hand, the class labeled 2 ("Speed limit (50km/h)") has 2010 examples, which is over an order of magnitude difference!

<img src="./examples/class-2-examples.png" />

This begs the question, why the huge inequality? Is a driver in Germany really 10 times as likely to encounter a 50 km/h sign vs. a 20 km/h sign? However, I ultimately chose to keep leave that distribution unchanged, as it closely matches the dataset as a whole.

---
## Designing and Testing a Convolutional Neural Net Model

Prior to this project, I built a [Lenet-5](http://yann.lecun.com/exdb/lenet/) implementation. This used the following model architecture.

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x3 RGB image                             |
| Convolution 1 5x5     | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 2 5x5     | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x16     |
| Flatten               | outputs 400                                   |
| Fully connected       | outputs 120                                   |
| RELU                  |                                               |
| Fully connected       | outputs 84                                    |
| RELU                  |                                               |
| Fully connected       | outputs 43 (matching the number of classes)   |
| Softmax Cross Entropy | Reduce mean. Adam Optimizer                   |

Training the data on this model yielded ~89% accuracy on the validation set and a little over 90% on the test set. Not terrible result for a first pass, but could definitely be improved upon.

### Preprocessing

<img src="./examples/rand-color-0.png" />

Above is a random sampling of images before preprocessing. It's easy to see that some of these images are very dark and would be challenging for even a human to classify. So, the first step I took to improve this result was to preprocess the data in an attempt to better amplify the important features in the image.

I experimented with a number of ways to process and normalize the data, such as running the images through a Laplacian filter to enhance the edges in the images and normalize the values. However, the model performed best with histogram equalization and normalizing the image values from -0.5 to 0.5. It is also worth noting that there was no apparent degradation in performance when converting the input images to a single grayscale channel from RGB.

<img src="./examples/rand-gray-0.png" />

By applying histogram equalization to the images, which is conveniently handled by OpenCV's `equalizeHist` method, it is possible to correct for very dark and very bright lighting conditions. With just the preprocessing step, the model was able to achieve ~92% accuracy on the validation set.

### Improved Model Architecture

In order to achieve better performance, I needed to build a better CNN model. I decided to build the model described by [Alex Staravoitau](http://navoshta.com/traffic-signs-classification/#model). This differs from a simple, strictly feed-forward model like LeNet-5 in that the convolved, pooled layers are branched off and fed into a flattened, fully connected layer, noting that each of these convolutions is first passed through an additional max-pooling proportional to their layer size. In order to prevent overfitting to the training data, I also applied a 50% dropout to the fully connected layers.

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 Grayscale image                       |
| Convolution 1 5x5     | 1x1 stride, same padding, outputs 32x32x32    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, same padding, outputs 16x16x32    |
| Convolution 2 5x5     | 1x1 stride, same padding, outputs 16x16x64    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, same padding, outputs 8x8x64      |
| Convolution 3 5x5     | 1x1 stride, same padding, outputs 8x8x128     |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, same padding, outputs 4x4x128     |
| Flatten               | flatten each of the 3 pooled layers.          |
| Max pooling - layer 1 | 4x4 stride, same padding, outputs 1x2048      |
| Max pooling - layer 2 | 2x2 stride, same padding, outputs 1x1024      |
| Max pooling - layer 3 | 2x2 stride, same padding, outputs 1x512       |
|                       | Concatenate the 3 pooled layers, outputs 1x3584 |
| 50% dropout           |                                               |
| Fully connected 1     | outputs 1x1024                                |
| 50% dropout           |                                               |
| Fully connected 2     | outputs 1x43                                  |
| Softmax Cross Entropy | Reduce mean. Adam Optimizer                   |

### Model Training

I decided on a simple learning rate of 0.001 over 100 epochs, using a batch size of 256. The weights were initialized with TensorFlow's `truncated_normal` method, with a mean of 0 and a standard deviation of 0.1. The loss was calculated by applying a softmax cross entropy function, comparing the predicted classes with the validation set. This is then optimized with the `tf.train.AdamOptimizer`, which uses Kingma and Ba's [Adam algorithm](https://arxiv.org/pdf/1412.6980v8.pdf) for first-order gradient-based optimization of randomized objective functions. Adam enables the model to use a large step size and move the model quickly to convergence without a lot of fine tuning.

And indeed, the accuracy of this model converges very quickly, with very little improvement seen beyond ~20 epochs.

My final model results were:
* validation set accuracy of 97%
* test set accuracy of 95.5%

### Testing the Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image              |     Prediction                    |
|:---------------------:|:---------------------------------------------:|
| Stop Sign          | Stop sign                     |
| U-turn           | U-turn                     |
| Yield          | Yield                      |
| 100 km/h            | Bumpy Road                   |
| Slippery Road      | Slippery Road                    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           |     Prediction                    |
|:---------------------:|:---------------------------------------------:|
| .60               | Stop sign                     |
| .20             | U-turn                     |
| .05          | Yield                      |
| .04              | Bumpy Road                   |
| .01            | Slippery Road                    |


For the second image ...
