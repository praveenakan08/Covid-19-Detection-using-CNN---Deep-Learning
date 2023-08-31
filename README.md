# Covid-19 Prediction using CNN - Deep Learning

The project's name is Detecting Covid-19 disease using DL.
CNN -> Convolutional Neural Network is generally used for Image Recognition, Image Classification, Object Detection, and Ibject Segmentation.
	It has layers such as, the Pooling layer which can be max or average, Convultional layer, Recitfied Linear layer, Sigmoid layer (used as the putput layer),
Flatenning Layer, Fully Connected Layer.

 -> Input layer accepts the pixels of image in the form of arrays.
 -> Hidden layers carry out feature extraction by performing certain calculations and manipulations to detect patterns in the image.
 -> Output layer is fully connected layer that identifies the object in the image.
 
Convolutional Layer:	a filter matrix is filled with features in the image and it is slided throughout the image and dot product is computed to detect patterns.
Relu Layer:	performs element wise operations, sets all -ve values to 0.
Pooling layer:	is a down sampling operation that reduces the dimensionality of the feature map and increases the intensity of the pixels in the image.
Flatenning:	converting all the resultant 2D arrays from pooled feature map into a single long continuous linear vector.
Fully connected:	flattened matrix -> given to the op layer to classify the image.

For this project, we have collected a data set of approximately 700 CT scan images of chests of various Covid-19 +ve and normal patients and 
divided the dataset into train data and test data with 7:3 ratio.

Then, we created an image generator for generating new data which is called Data augmentation using ImageGenerator. 
That was our part of pre-processing. 

We used GPU which was available on the Kaggle platform which provides us an instance of the GPU for faster computation.

Then a standard CNN architecture was used, called VGG-16, because these standard architectures were proven to give accurate results in the past.

We tried using other archiectures also such as ResNet, AlexNet, and LeNet as part of hyper-parameter tuning but those don't seem to give good results 
with our data set so, we chose VGG-16 Net.

VGG-16:	CNN layers are stacked up with increasing filter sizes that is if layer-1 has 16 filters, then layer-2 must have 16 or more filters
	-> All filters are of sizez 3*3
	-> 2 3*3 filters almost cover the area of what a 5*5 filter covers and cheaper in computing than 5*5 
 
image shape - 224,224,3
model -> sequential

Model Architecture:
2 conv+maxPool2d(padding=same,relu,64) -> 2conv+maxPpool(128)+conv(same, relu,256) -> 2conv+MaxPool(256,same, relu) -> 3conv+max(same,relu,512) -> 3convd+maxpool(512,same, relu) -> flatten -> 2dense(4096units,relu) -> dense(2,softmax)

The softmax layer turns smaller or negative values into lesser probablitity and large values into higher probabilities.

Then for the optimization part, we used Stochastic Gradient Descent optimizer (SGD), tried using other optimizers (Adam, GD, RSProp), 
with learning rate 0.01 and catergoriacal crossentropy as the loss function.

Then we fitted the model to the traindata and took validation data as test data with 64 epochs and 12 steps per epoch
epochs-> no. of times the whole dataset is trained
steps_per_epoch-> batches of samples to train, how many batches of samples to use in one epoch

This gave us the train accuracy of 83% and test accuracy of about 79%.
