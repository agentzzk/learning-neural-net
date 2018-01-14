# Get libraries
# pip install <theano, tensorflow, keras>

# import libraries to build the CNN
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

# Setup and initialize the CNN
classifier = Sequential()
inputShape = (64, 64, 3)

# Step 1 - Convolutional layer: generate 32 feature maps using 3x3 kernels; rectifier activation
classifier.add(Conv2D(32, (3, 3), padding = 'same', input_shape = inputShape))
classifier.add(Activation('relu'))

# Step 2 - Pooling layer: reduce the data size using 2x2 max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flatten the data to be fed into the CNN
classifier.add(Flatten())

# Step 4 - Define the CNN model to use with 128 inputs
classifier.add(Dense(128))
classifier.add(Activation("relu"))

# Step 4 - Add output layer to CNN model; classify as binary result
classifier.add(Dense(1))
classifier.add(Activation("softmax"))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator

# transform images to increase training set; improve accuracy
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# train the data set
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=200)