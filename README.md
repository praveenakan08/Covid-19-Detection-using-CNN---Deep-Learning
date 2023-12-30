# Covid-19 Prediction using CNN - Deep Learning

# Data Collection:
For this project, we have collected a data set of approximately 700 CT scan images of chests of various Covid-19 +ve and normal patients and 
divided the dataset into train data and test data with 7:3 ratio.

# Data Pre-processing:
Then, as part of Data pre-processing, we created an image generator for generating new data which is called Data augmentation using ImageGenerator.  

We used GPU which was available on the Kaggle platform which provides us an instance of the GPU for faster computation.

# Model Building: 
VGG-16 architecture was used as our base model. We experimented with other archiectures also such as ResNet, AlexNet, and LeNet as part of hyper-parameter tuning but, VGG-16 gave the best results with the dataset we used.

# Our Model Architecture:
-> 2 Convolutional layers + MaxPool2D (padding = same, activation function = relu, input units = 64) 

-> 2 convolutional layers + MaxPool2D (padding = same, activation function = relu, input units = 128) 

-> 2 convolutional layers + MaxPool2D (padding = same, activation function = relu, input units = 256) 

-> 3 convolutional layers + MaxPool2D (padding = same,activation function = relu, input units = 512)  

-> Flattening layer 

-> 2 Dense layers (activation function = relu, input units = 4096) 

-> Dense layer (activation function = softmax, input units = 2)

# Optimization: 
We used Stochastic Gradient Descent optimizer (SGD), experimented with other optimizers such as Adam, Gradient Descent and RSProp, 
with learning rate 0.01 and categoriacal crossentropy as the loss function.

# Model Training:
The model was trained with train Data and validation data was used for testing the performance of the model for 64 epochs and 12 steps per epoch.
This model resulted in a classfication accuracy of 83%.
