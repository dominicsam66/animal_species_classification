# Libraries used
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt # to plot the training and validation accuracy and loss

# Applying Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Define the directory for images
train_dir = r'C:\my_drive\ml_dl_database\Data_for_CNN_Copy\Train'
val_dir = r'C:\my_drive\ml_dl_database\Data_for_CNN_Copy\Val'
test_dir = r'C:\my_drive\ml_dl_database\Data_for_CNN_Copy\Test'

#Batch size
bt=16 # Batch size for training data
bv=16 # Batch size for validation data

#Value for L2 Regulaizer
ltwo=1

# Learning rate
lr=0.00001

# Create training, validation, and testing data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=bt,
    class_mode='categorical')

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=bv,
    class_mode='categorical')

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical')

# Display class indices of training dat
print(train_generator.class_indices)  # Each folder (species) will be assigned a class label

# Get a batch of images and labels for training
train_images, train_labels = next(train_generator)

# Print the shape of the batch (16 images in this example) for training
print("training batch size:", train_images.shape)  # e.g., (16, 150, 150, 3)

# Print the labels (these will be one-hot encoded) for training
print("Labels for the training batch (one-hot encoded):", train_labels)

# Convert one-hot encoded labels back to class indices (e.g., 0, 1, 2) for training
train_class_indices = train_labels.argmax(axis=1)
print("training class indices for the batch:", train_class_indices)

# Display class indices of validation data
print(validation_generator.class_indices)  # Each folder (species) will be assigned a class label

# Get a batch of images and labels for validation
validation_images, validation_labels = next(validation_generator)

# Print the shape of the batch (16 images in this example) for validation
print("validation batch size:", validation_images.shape)  # e.g., (32, 150, 150, 3)

# Print the labels (these will be one-hot encoded) for validation
print("Labels for the validation batch (one-hot encoded):", validation_labels)

# Convert one-hot encoded labels back to class indices (e.g., 0, 1, 2) for validation
validation_class_indices = validation_labels.argmax(axis=1)
print("validation class indices for the batch:", validation_class_indices)

# Build the CNN model
model = Sequential()

model.add(Input(shape=(150, 150, 3)))  # Correct way to define input shape, Input size is based on image size

# First Hidden Layer
model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(ltwo)))#convolution layer
model.add(MaxPooling2D(pool_size=(2, 2)))#Max pooling layer

# Second Hidden Layer
model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(ltwo))) #convolution layer
model.add(MaxPooling2D(pool_size=(2, 2)))#Max pooling layer

# Third Hidden Layer
model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(ltwo))) #convolution layer
model.add(MaxPooling2D(pool_size=(2, 2)))#Max pooling layer

# Flatten the output
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(32, activation='relu', kernel_regularizer=l2(ltwo)))
model.add(Dropout(0.5)) # First Dropout layer to reduce overfitting
model.add(Dropout(0.3)) # Second Dropout layer to reduce overfitting


# Output Layer (assuming 10 species)
model.add(Dense(3, activation='softmax')) # output layer for classification

# Compile the model
optimizer = Adam(learning_rate=lr)  # optimizer
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) #compiling the model

# Train the model
history = model.fit(train_images,train_labels, batch_size=bt,epochs=1500, validation_data=(validation_images,validation_labels),validation_batch_size=bv) #Fiiting the model

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=bv)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_acc}')