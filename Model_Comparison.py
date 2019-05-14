# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:47:14 2019

@author: Will's PC
"""
'''
Model Comparison
# Run after main CNN so that target variables in workspace
'''
############################################################################
# INPUTS
import tensorflow as tf
import numpy as np
from keras import applications
from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import regularizers
import scipy.io as sio
from keras.callbacks import ModelCheckpoint 
import matplotlib.pyplot as plt

############################################################################
# VGG-16 Model

print('Training VGG-16')

train_data = np.load(open('Bottleneck_Features\VGG16_bottleneck_features_train.npy','rb'))
valid_data = np.load(open('Bottleneck_Features\VGG16_bottleneck_features_validation.npy','rb'))
test_data = np.load(open('Bottleneck_Features\VGG16_bottleneck_features_test.npy','rb'))



VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
#VGG16_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
VGG16_model.add(Dense(120, activation='softmax'))

VGG16_model.summary()


VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16 = VGG16_model.fit(train_data, train_targets, 
          validation_data=(valid_data,  validation_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_data]


VGG16_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('VGG16 accuracy: %.4f%%' % VGG16_accuracy)


############################################################################
# VGG-19 Model

print('Training VGG-19')

train_data = np.load(open('Bottleneck_Features\VGG19_bottleneck_features_train.npy','rb'))
valid_data = np.load(open('Bottleneck_Features\VGG19_bottleneck_features_validation.npy','rb'))
test_data = np.load(open('Bottleneck_Features\VGG19_bottleneck_features_test.npy','rb'))



VGG19_model = Sequential()
VGG19_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
VGG19_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
VGG19_model.add(Dropout(0.4))
VGG19_model.add(Dense(120, activation='softmax'))

VGG19_model.summary()


VGG19_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG19.hdf5', 
                               verbose=1, save_best_only=True)

VGG19 = VGG19_model.fit(train_data, train_targets, 
          validation_data=(valid_data,  validation_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


VGG19_predictions = [np.argmax(VGG19_model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_data]


VGG19_accuracy = 100*np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
print('VGG19 accuracy: %.4f%%' % VGG19_accuracy)

############################################################################
# ResNet-50 Model

print('Training ResNet-50')

train_data = np.load(open('Bottleneck_Features\ResNet50_bottleneck_features_train.npy','rb'))
valid_data = np.load(open('Bottleneck_Features\ResNet50_bottleneck_features_validation.npy','rb'))
test_data = np.load(open('Bottleneck_Features\ResNet50_bottleneck_features_test.npy','rb'))



ResNet50_model = Sequential()
ResNet50_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
ResNet50_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
ResNet50_model.add(Dropout(0.4))
ResNet50_model.add(Dense(120, activation='softmax'))

ResNet50_model.summary()


ResNet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.ResNet50.hdf5', 
                               verbose=1, save_best_only=True)

ResNet50 = ResNet50_model.fit(train_data, train_targets, 
          validation_data=(valid_data,  validation_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


ResNet50_predictions = [np.argmax(ResNet50_model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_data]


ResNet50_accuracy = 100*np.sum(np.array(ResNet50_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('ResNet-50 accuracy: %.4f%%' % ResNet50_accuracy)

############################################################################
# InceptionV3 Model

print('Training InceptioV3')

train_data = np.load(open('Bottleneck_Features\InceptionV3_bottleneck_features_train.npy','rb'))
valid_data = np.load(open('Bottleneck_Features\InceptionV3_bottleneck_features_validation.npy','rb'))
test_data = np.load(open('Bottleneck_Features\InceptionV3_bottleneck_features_test.npy','rb'))

train_data.shape[1:]

inception_model = Sequential()
inception_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
inception_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
inception_model.add(Dropout(0.4))
inception_model.add(Dense(120, activation='softmax'))

inception_model.summary()

inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

inception_checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5', 
                               verbose=1, save_best_only=True)

inception = inception_model.fit(train_data, train_targets, 
          validation_data=(valid_data, validation_targets),
          epochs=20, batch_size=20, callbacks=[inception_checkpointer], verbose=1)

inception_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')

InceptionV3_predictions = [np.argmax(inception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_data]

InceptioV3_accuracy = 100*np.sum(np.array(InceptionV3_predictions)==np.argmax(test_targets, axis=1))/len(InceptionV3_predictions)

print('InceptioV3 accuracy: %.4f%%' % InceptioV3_accuracy)


############################################################################
# Xception Model
print('Training Xception')

train_data = np.load(open('Bottleneck_Features\Xception_bottleneck_features_train.npy','rb'))
valid_data = np.load(open('Bottleneck_Features\Xception_bottleneck_features_validation.npy','rb'))
test_data = np.load(open('Bottleneck_Features\Xception_bottleneck_features_test.npy','rb'))

train_data.shape[1:]

Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
Xception_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
Xception_model.add(Dropout(0.4))
Xception_model.add(Dense(120, activation='softmax'))

Xception_model.summary()

Xception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

Xception_checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5', 
                               verbose=1, save_best_only=True)

Xception = Xception_model.fit(train_data, train_targets, 
          validation_data=(valid_data, validation_targets),
          epochs=20, batch_size=20, callbacks=[Xception_checkpointer], verbose=1)

Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')

Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_data]

Xception_accuracy = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/len(Xception_predictions)

print('Test accuracy: %.4f%%' % Xception_accuracy)

############################################################################
# MobileNet Model

print('Training MobileNet')
train_data = np.load(open('Bottleneck_Features\MobileNet_bottleneck_features_train.npy','rb'))
valid_data = np.load(open('Bottleneck_Features\MobileNet_bottleneck_features_validation.npy','rb'))
test_data = np.load(open('Bottleneck_Features\MobileNet_bottleneck_features_test.npy','rb'))

train_data.shape[1:]

MobileNet_model = Sequential()
MobileNet_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
MobileNet_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
MobileNet_model.add(Dropout(0.4))
MobileNet_model.add(Dense(120, activation='softmax'))

MobileNet_model.summary()

MobileNet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

MobileNet_checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.MobileNet.hdf5', 
                               verbose=1, save_best_only=True)

MobileNet = MobileNet_model.fit(train_data, train_targets, 
          validation_data=(valid_data, validation_targets),
          epochs=20, batch_size=20, callbacks=[MobileNet_checkpointer], verbose=1)

MobileNet_model.load_weights('saved_models/weights.best.MobileNet.hdf5')

MobileNet_predictions = [np.argmax(MobileNet_model.predict(np.expand_dims(feature, axis=0))) for feature in test_data]

MobileNet_accuracy = 100*np.sum(np.array(MobileNet_predictions)==np.argmax(test_targets, axis=1))/len(MobileNet_predictions)

print('MobileNet accuracy: %.4f%%' % MobileNet_accuracy)
print('VGG16 accuracy: %.4f%%' % VGG16_accuracy)
print('VGG19 accuracy: %.4f%%' % VGG19_accuracy)
print('ResNet-50 accuracy: %.4f%%' % ResNet50_accuracy)
print('InceptioV3 accuracy: %.4f%%' % InceptioV3_accuracy)
print('Xception accuracy: %.4f%%' % Xception_accuracy)



# PLOT RESULTS
plt.plot(VGG16.history['val_acc'])
plt.plot(VGG19.history['val_acc'])
plt.plot(ResNet50.history['val_acc'])
plt.plot(inception.history['val_acc'])
plt.plot(Xception.history['val_acc'])
plt.plot(MobileNet.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['VGG16', 'VGG19','ResNet50','Xception','inception','MobileNet'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(VGG16.history['val_loss'])
plt.plot(VGG19.history['val_loss'])
plt.plot(ResNet50.history['val_loss'])
plt.plot(inception.history['val_loss'])
plt.plot(Xception.history['val_loss'])
plt.plot(MobileNet.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['VGG16', 'VGG19','ResNet50','Xception','inception','MobileNet'], loc='upper left')
plt.show()













