#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* Experiment_1.ipynb and Experiment_2.ipynb Jupyter notebooks detailing some
  initial experiments with the problem

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution
neural network. The file shows the pipeline I used for training and validating
the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model archtiecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes
and depths between 24 and 64 (model.py lines 87-105).
 
The model includes RELU layers to introduce nonlinearity.

####2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py lines
99).

The model was trained and validated on different data sets to ensure that the
model was not overfitting. The model was tested by running it through the
simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually
(model.py line 72).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a
combination of the sample data provided in the project resources and two sets of
recovery images that I recorded myself.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to mimic the NVidia
architecture described in
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf. The
main difference to the NVidia is the input image resolution (80x40 vs 200x600).

My first step was to use a convolution neural network model similar to the
LeNet. I thought this model might be appropriate because the LeNet architecture
has been found to perform well in image-based learning tasks. The initial
experiment with LeNet is detailed in Experiement_1.ipynb, where I found the
model could overfit the three hand-picked training images.

I continued working with the LeNet in Experiment_2.ipynb, where I tried the
model with the whole Udacity driving data. I did not seem to get the training to
produce low validation error, so I turned to the NVidia architecture, which did
not seem to perform much better.

At this point I was somewhat perplexed how to proceed, but after reading some
tips from other people who had completed the project I decided to just start
testing the model with the simulator. After quite a lot of tweaking and trying
different tips, I managed to get the car to drive around the track
successfully. 

At hindsight the main "trick" that helped me the most was to scaling the
steering angles in the training to range [-0.15,0.15]. I recorded the recovery
data using a keyboard, which resulted to mostly -1, 0, 1 steering
angles. Learning with this data resulted to models that tended drive in very
zig-zag fashion that caused a lot of cases where the car veers off the
road. After scaling the steering angles the car started to behave in much more
smooth fashion.

Another helpful trick was to save the model with the lowest validation score
using a callback in the model.fit() function. One of these intermediate models
turned out to be the one that managed to drive around the track for the first
time. This model still has a tendency to oversteer when it approaches the road's
edge, but it usually returns to driving straight after a while. 

One of the main tips from the forums was to use the left and right images from
the driving data. I implemented this but in the end I did not have to use that
to train the working model.

####2. Final Model Architecture

The final model architecture (model.py lines 87-105) consisted of a convolution
neural network with the following layers and layer sizes:

* Layer  1 convolution2d_1  input shape (None, 40, 80, 3) output shape (None, 36, 76, 24)
* Layer  2 activation_1     input shape (None, 36, 76, 24) output shape (None, 36, 76, 24)
* Layer  3 convolution2d_2  input shape (None, 36, 76, 24) output shape (None, 16, 36, 36)
* Layer  4 activation_2     input shape (None, 16, 36, 36) output shape (None, 16, 36, 36)
* Layer  5 convolution2d_3  input shape (None, 16, 36, 36) output shape (None, 6, 16, 48)
* Layer  6 activation_3     input shape (None, 6, 16, 48) output shape (None, 6, 16, 48)
* Layer  7 convolution2d_4  input shape (None, 6, 16, 48) output shape (None, 4, 14, 64)
* Layer  8 activation_4     input shape (None, 4, 14, 64) output shape (None, 4, 14, 64)
* Layer  9 convolution2d_5  input shape (None, 4, 14, 64) output shape (None, 2, 12, 64)
* Layer 10 activation_5     input shape (None, 2, 12, 64) output shape (None, 2, 12, 64)
* Layer 11 flatten_1        input shape (None, 2, 12, 64) output shape (None, 1536)
* Layer 12 dense_1          input shape (None, 1536) output shape (None, 100)
* Layer 13 dropout_1        input shape (None, 100) output shape (None, 100)
* Layer 14 activation_6     input shape (None, 100) output shape (None, 100)
* Layer 15 dense_2          input shape (None, 100) output shape (None, 50)
* Layer 16 activation_7     input shape (None, 50) output shape (None, 50)
* Layer 17 dense_3          input shape (None, 50) output shape (None, 10)
* Layer 18 activation_8     input shape (None, 10) output shape (None, 10)
* Layer 19 dense_4          input shape (None, 10) output shape (None, 1)

####3. Creation of the Training Set & Training Process

As the basis for the training data I used the Udacity's sample driving data from
the project resources. I then recorded the vehicle recovering from the left side
and right sides of the road back to center. The first recovery set was ~969
images, but I decided to do another set that contained 1586 additional images.

Here is an example of center lane driving from the Udacity driving data:

![alt text][./report_images/center_2016_12_01_13_31_13_037.jpg]

Here are couple images from the recovery sets:

![alt text][./report_images/center_2017_02_07_19_00_02_859.jpg]
![alt text][./report_images/right_2017_02_07_19_40_42_810.jpg]

After the collection process, I had 10591 number of data points.

The simulator images were scaled down from 320x160 to 80x40 for training. The
same scaling is also included in drive.py.

I let the model.fit randomly shuffle the data set and put 0.2 of the data into a
validation set.

I used this training data for training the model. The validation set helped
determine if the model was over or under fitting. The ideal number of epochs was
around 3 to 5 as evidenced by the final model. I used an Adam optimizer so that
manually training the learning rate wasn't necessary.
