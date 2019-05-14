# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:03:39 2019

@author: William Norman
"""
# Import Libraries 
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.utils import to_categorical 
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import functools

'''
Initialise Targets
'''
# Load the image annotations and lacations indexes
file_mat = sio.loadmat('file_list.mat')
train_mat = sio.loadmat('train_list.mat')
test_mat = sio.loadmat('test_list.mat')
annotations_mat = sio.loadmat('test_list.mat')
Classes = np.load('Classes.npy')


# Initialise the class targets 
train_targets = []
test_targets = []
validation_targets = []

# Loop through all labels and add label to target array
for x in range (len(train_mat['file_list'])):
    train_targets = np.append(train_targets, train_mat['labels'][x][0]-1)
        
for x in range (0,len(test_mat['file_list'])-1,2):
    test_targets = np.append(test_targets, test_mat['labels'][x][0]-1)
    
for x in range (1,len(test_mat['file_list'])-1,2):
    validation_targets = np.append(validation_targets, test_mat['labels'][x][0]-1)

# One-Hot Encode label targets
train_targets = to_categorical(train_targets) 
test_targets = to_categorical(test_targets) 
validation_targets = to_categorical( validation_targets) 


'''
Initial Scratch Model
'''
# Set image dimension
image_size = 250

# Initialise Datagen
datagen = ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')

# Setup sequential network
model = Sequential()
model.add(Conv2D(filters = 16,kernel_size = (5,5),strides = (2,2),padding = 'valid', activation ='relu',input_shape = (image_size, image_size, 3)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), strides = (4,4), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid'))
model.add(Conv2D(filters = 64, kernel_size = (2,2), strides = (2,2), padding = 'valid', activation = 'relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(120, activation='softmax'))
model.summary()

# Compile model using Adam optimiser
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

# Set batch size
batch_size = 600



# Initialise Test Datagen 
test_datagen = ImageDataGenerator(rescale=1./255)

# Augment training images
train_generator = datagen.flow_from_directory(
        'Annotation/Training',  
        target_size=(image_size, image_size), 
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = True)  

# Augment validation images
validation_generator = test_datagen.flow_from_directory(
        'Annotation/Validation',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = True)

# Save the model parameters
Initial_checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Initial.hdf5', 
                               verbose=1, save_best_only=True)

# Train the network using the generators 
history = model.fit_generator(
                train_generator,
                steps_per_epoch= 12000 // batch_size,
                epochs=20,
                validation_data=validation_generator,
                validation_steps= 4289 // batch_size,
                callbacks=[Initial_checkpointer])

# Save the final weights 
model.save_weights('Initial_try.h5')  # always save your weights after training or during training

# Plot results of initial model 
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



'''
PERFORMANCE
'''
# Initialise test datagen
test_generator = test_datagen.flow_from_directory('annotation/Testing', 
                                                    target_size=(image_size, image_size), 
                                                    batch_size=1,
                                                    class_mode='categorical',
                                                    shuffle=False)
# Reset the test generator
test_generator.reset()
# Predict class on the test set
predictions = model.predict_generator(test_generator,4290) 
# Find the class value
Initial_predictions = predictions.argmax(axis=1) 
# Summarise the overall accuracy of the model
Initial_accuracy = 100*np.sum(np.array(Initial_predictions)==np.argmax(test_targets, axis=1))/len(Initial_predictions)
# Print the final accuracy
print('Test accuracy: %.4f%%' % Initial_accuracy)    


'''
TRANSFER LEARNOMG Using InceptionV3
'''
# Load the Bottleneck Features from the Xception model
train_data = np.load(open('Bottleneck_Features\Xception_bottleneck_features_train.npy','rb'))
valid_data = np.load(open('Bottleneck_Features\Xception_bottleneck_features_validation.npy','rb'))
test_data = np.load(open('Bottleneck_Features\Xception_bottleneck_features_test.npy','rb'))

# Initialise Transfer Model
train_data.shape[1:]
Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
Xception_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
Xception_model.add(Dropout(0.4))
Xception_model.add(Dense(120, activation='softmax'))
# Sumarise the model
Xception_model.summary()
# Set up top 3 accuracy metric
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
# Compile model
Xception_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy', top3_acc])
# Save Xception weights to memory
InceptionV3_checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5', 
                               verbose=1, save_best_only=True)
# Train the Transfer Model Parameters
history = Xception_model.fit(train_data, train_targets, 
          validation_data=(valid_data, validation_targets),
          epochs=25, batch_size=300, callbacks=[InceptionV3_checkpointer], verbose=1)
# Load the saved weights
Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')
# Calculate Model Accuracy
Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_data]
test_accuracy = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/len(Xception_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# PLOT RESULTS
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['top3_acc'])
plt.plot(history.history['val_top3_acc'])
plt.title('Top K Accuracy (K=5)')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()








