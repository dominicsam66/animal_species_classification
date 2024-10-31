# animal_species_classification

A neural network is developed to classify different species of animals based on the image. For achieving this task, convolution neural network was employed using keras and tensorflow. A base model was created in the beginning and resulted in overfitting. To reduce overfitting, strategies like data augmentation, L2 regularization, additional hidden layer and dropout layers were used. Using these overfitting was reduced significantly.

Step 1:- Import all the necessary libraries like models, layers, TensorFlow, Keras and matplotlib.

Step 2:- Training, validation and testing data are imported using 'flow_from_directory' function. Data augmentation was done on training data using transformations like horizontal flipping, zooming, shear and rescaling pixel values to [0 1].

Step 3:- Training, validation and testing data are labelled based on the subfolders. Each subfolder has only images of  one species.

Step 4:- Model was build based on a architecture consisting of input layer, convolutional layer, max pooling layer, flatten layer, fully connected layer and output layer.
	The CNN architecture consists of:
1) Input Layer: Accepts images of size 150x150 with 3 color channels (RGB).
2) Convolutional Layers: Three convolutional layers with ReLU activation and L2 regularization.
3) Max Pooling Layers: Following each convolutional layer to reduce spatial dimensions.
4) Flatten Layer: Converts 2D matrices into 1D vectors.
5) Fully Connected Layer: A dense layer with ReLU activation and dropout for regularization.
6) Output Layer: Softmax activation for multi-class classification

Step 5:- The model is compiled using the Adam optimizer with a learning rate of 0.00001 and trained for 1500 epochs on the training dataset, while validating on the validation dataset.

Step 6:- Training and validation accuracy and loss are plotted using Matplotlib for visual analysis of the model's performance over epochs.

Step 7:- The model's performance is evaluated on the validation set, printing out both loss and accuracy metrics.
