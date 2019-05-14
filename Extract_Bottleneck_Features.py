# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:48:26 2019

@author: Will's PC
"""
import tensorflow as tf
import numpy as np
from keras import applications
from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense,GlobalAveragePooling2D
from keras.optimizers import Adam
import scipy.io as sio
from keras.callbacks import ModelCheckpoint 


datagen = ImageDataGenerator(rescale=1. / 255)

img_width = 200
img_height = 200
Batch_Size  = 1


file_mat = sio.loadmat('file_list.mat')
train_mat = sio.loadmat('train_list.mat')
test_mat = sio.loadmat('test_list.mat')
annotations_mat = sio.loadmat('test_list.mat')

'''
########################################################################
Initialise Data Generators 
'''


train_generator = datagen.flow_from_directory('annotation/Training', 
                                                    target_size=(img_height, img_width), 
                                                    batch_size=Batch_Size,
                                                    class_mode='categorical',
                                                    shuffle=False)
                                                    

test_generator = datagen.flow_from_directory('annotation/Testing', 
                                                    target_size=(img_height, img_width), 
                                                    batch_size=Batch_Size,
                                                    class_mode='categorical',
                                                    shuffle=False)

validation_generator = datagen.flow_from_directory('annotation/Validation', 
                                                    target_size=(img_height, img_width), 
                                                    batch_size=Batch_Size,
                                                    class_mode='categorical',
                                                    shuffle=False)

'''
########################################################################
Load VGG16 Model and throw away top layer, then save bottleneck features
'''
print('Extracting VGG-16 Bottleneck Features')

model = applications.VGG16(include_top=False, weights='imagenet')

bottleneck_features_train = model.predict_generator(
        train_generator,12000) 

np.save(open('Bottleneck_Features/VGG16_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


bottleneck_features_validation = model.predict_generator(
        validation_generator,4289) 

np.save(open('Bottleneck_Features/VGG16_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

bottleneck_features_test = model.predict_generator(
        test_generator,4290) 

np.save(open('Bottleneck_Features/VGG16_bottleneck_features_test.npy', 'wb'), bottleneck_features_test)


'''
train_VGG16 = np.load(open('Bottleneck_Features\VGG16_bottleneck_features_train.npy','rb'))
valid_VGG16 = np.load(open('Bottleneck_Features\VGG16_bottleneck_features_validation.npy','rb'))
test_VGG16 = np.load(open('Bottleneck_Features\VGG16_bottleneck_features_test.npy','rb'))



VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
#VGG16_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
VGG16_model.add(Dense(120, activation='softmax'))

VGG16_model.summary()


VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16,  validation_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_VGG16]


test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
'''


#########################################################################
# Extract Bottleneck features for VGG-19
print('Extracting VGG-19 Bottleneck Features','\n',)


model = applications.VGG19(include_top=False, weights='imagenet')

bottleneck_features_train = model.predict_generator(
        train_generator,12000) 

np.save(open('Bottleneck_Features/VGG19_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


bottleneck_features_validation = model.predict_generator(
        validation_generator,4289) 

np.save(open('Bottleneck_Features/VGG19_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

bottleneck_features_test = model.predict_generator(
        test_generator,4290) 

np.save(open('Bottleneck_Features/VGG19_bottleneck_features_test.npy', 'wb'), bottleneck_features_test)



########################################################################
#Extract Bottleneck features for ResNet-50
print('\n','Extracting ResNet Bottleneck Features','\n')

model = applications.ResNet50(include_top=False, weights='imagenet')

bottleneck_features_train = model.predict_generator(
        train_generator,12000) 

np.save(open('Bottleneck_Features/ResNet50_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


bottleneck_features_validation = model.predict_generator(
        validation_generator,4289) 

np.save(open('Bottleneck_Features/ResNet50_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

bottleneck_features_test = model.predict_generator(
        test_generator,4290) 

np.save(open('Bottleneck_Features/ResNet50_bottleneck_features_test.npy', 'wb'), bottleneck_features_test)


########################################################################
#Extract Bottleneck features for InceptionV3
print('\n','Extracting InceptionV3 Bottleneck Features')

model = applications.InceptionV3(include_top=False, weights='imagenet')

bottleneck_features_train = model.predict_generator(
        train_generator,12000) 

np.save(open('Bottleneck_Features/InceptionV3_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


bottleneck_features_validation = model.predict_generator(
        validation_generator,4289) 

np.save(open('Bottleneck_Features/InceptionV3_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

bottleneck_features_test = model.predict_generator(
        test_generator,4290) 

np.save(open('Bottleneck_Features/InceptionV3_bottleneck_features_test.npy', 'wb'), bottleneck_features_test)


########################################################################
#Extract Bottleneck features for Xception
from keras import applications

print('\n','Extracting Xception Bottleneck Features')

model = applications.Xception(include_top=False, weights='imagenet')

bottleneck_features_train = model.predict_generator(
        train_generator,12000) 

np.save(open('Bottleneck_Features/Xception_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


bottleneck_features_validation = model.predict_generator(
        validation_generator,4289) 

np.save(open('Bottleneck_Features/Xception_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

bottleneck_features_test = model.predict_generator(
        test_generator,4290) 

np.save(open('Bottleneck_Features/Xception_bottleneck_features_test.npy', 'wb'), bottleneck_features_test)


########################################################################
#Extract Bottleneck features for MobileNet
print('\n','Extracting MobileNet Bottleneck Features')

model = applications.MobileNet(include_top=False, weights='imagenet')

bottleneck_features_train = model.predict_generator(
        train_generator,12000) 

np.save(open('Bottleneck_Features/MobileNet_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


bottleneck_features_validation = model.predict_generator(
        validation_generator,4289) 

np.save(open('Bottleneck_Features/MobileNet_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

bottleneck_features_test = model.predict_generator(
        test_generator,4290) 

np.save(open('Bottleneck_Features/MobileNet_bottleneck_features_test.npy', 'wb'), bottleneck_features_test)


########################################################################


