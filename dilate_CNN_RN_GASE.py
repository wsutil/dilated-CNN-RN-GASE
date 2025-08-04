from __future__ import print_function
import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Concatenate, Add, Maximum, Average
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, AveragePooling2D, LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import LearningRateScheduler
import math
import mat73
from scipy.stats import pearsonr
import scipy.io

batch_size = 128
epochs = 100

################################################################################
################################ LOAD DATA #####################################
################################################################################

dir_name = f"/path/to/sample_data/"
print(f"Directory name is: {dir_name}")

# load train data

fname = 'x_train_gase_scale.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
x_train_mat = mat73.loadmat(filename)['X_Train']
x_train_mat = np.float32(x_train_mat)

fname = 'y_train_gase_scale.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
y_train_mat = sio.loadmat(filename)['Y_Train']
y_train_mat = np.float32(y_train_mat)

# load test data
fname = 'x_test_gase_scale.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
x_test_mat = sio.loadmat(filename)['X_Test']
x_test_mat = np.float32(x_test_mat)

fname = 'y_test_gase_scale.mat'
filename = os.path.join(os.getcwd(),dir_name,fname)
y_test_mat = sio.loadmat(filename)['Y_Test']
y_test_mat = np.float32(y_test_mat)

print('x_train_shape:',x_train_mat.shape)
print('x_test_shape:',x_test_mat.shape)

print('y_train_shape:',y_train_mat.shape)
print('y_test_shape:',y_test_mat.shape)


# Form training and testing data
x_train = np.zeros((x_train_mat.shape[2],115,115,1),dtype=np.float32)
y_train = np.zeros((x_train_mat.shape[2],1),dtype=np.float32)

if x_test_mat.ndim ==2:
    x_test_mat =  x_test_mat.reshape((115, 115, 1))
    
x_test = np.zeros((x_test_mat.shape[2],115,115,1),dtype=np.float32)
y_test = np.zeros((x_test_mat.shape[2],1),dtype=np.float32)

for i in range(x_train_mat.shape[2]):
    temp = x_train_mat[:,:,i]
    temp = np.triu(temp,1)
    x_train[i,:,:,0] = temp
    y_train[i,0] = y_train_mat[i,0]
        
for i in range(x_test_mat.shape[2]):
    temp = x_test_mat[:,:,i]
    temp = np.triu(temp,1)
    x_test[i,:,:,0] = temp
    y_test[i,0] = y_test_mat[i,0]
    

##################################################################################
############################### DEFINE RN MODEL ##################################
##################################################################################

# Reshape input
print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

input_shape = (x_train.shape[1], x_train.shape[2], 1)

d_rate = 2
leaky_slope = 0.2

# Define 4 convolutional layers
def ConvolutionNetworks(no_filters=32, kernel_size=3, stride_size=1):
    def conv(model):
        model = Conv2D(no_filters, (3,3), strides=(stride_size,stride_size), activation='linear', input_shape=input_shape, data_format='channels_last')(model)
        model = LeakyReLU(negative_slope=leaky_slope)(model)
        model = MaxPooling2D()(model)
        model = BatchNormalization()(model)
        
        model = Conv2D(no_filters, (3,3), strides=(stride_size,stride_size), activation='linear')(model)
        model = LeakyReLU(negative_slope=leaky_slope)(model)
        model = MaxPooling2D()(model)
        model = BatchNormalization()(model)
        
        model = Conv2D(no_filters, (3,3), strides=(stride_size,stride_size), dilation_rate=(d_rate,d_rate), activation='linear')(model)
        model = LeakyReLU(negative_slope=leaky_slope)(model)
        model = MaxPooling2D()(model)
        model = BatchNormalization()(model)
        
        model = Conv2D(no_filters, (3,3), strides=(stride_size,stride_size), dilation_rate=(d_rate,d_rate), activation='linear')(model) # d=3
        model = LeakyReLU(negative_slope=leaky_slope)(model)
        model = MaxPooling2D()(model)
        model = BatchNormalization()(model)
        
        return model
    return conv


# Define function to compute relations from objects - the following uses just 4 Lambda layers - O(n^2) time complexity
def compute_relations(objects):
    
    def get_top_dim_1(t):
        return t[:,0,:,:]
    
    def get_all_but_top_dim_1(t):
        return t[:,1:,:,:]
    
    def get_top_dim_2(t):
        return t[:,0,:]
    
    def get_all_but_top_dim_2(t):
        return t[:,1:,:]
    
    slice_top_dim_1 = Lambda(get_top_dim_1)
    slice_all_but_top_dim_1 = Lambda(get_all_but_top_dim_1)
    slice_top_dim_2 = Lambda(get_top_dim_2)
    slice_all_but_top_dim_2 = Lambda(get_all_but_top_dim_2)
    
    d = K.int_shape(objects)[2]
    print('d = ', d)
    features = []
    
    for i in range(d):
        features1 = slice_top_dim_1(objects)
        objects = slice_all_but_top_dim_1(objects)
        
        for j in range(d):
            features2 = slice_top_dim_2(features1)
            features1 = slice_all_but_top_dim_2(features1)
            features.append(features2)
            
    relations = []
    concat = Concatenate()
    for feature1 in features:
        for feature2 in features:
            relations.append(concat([feature1,feature2]))
                    
            
    return relations

# Baseline model
def f_theta():
    def f(model):
        model = Dense(512)(model)
        model = LeakyReLU(negative_slope=leaky_slope)(model)
        model = Dropout(0.5)(model)
        
        model = Dense(512)(model)
        model = LeakyReLU(negative_slope=leaky_slope)(model)
        model = Dropout(0.5)(model)
        
        model = Dense(256)(model)
        model = LeakyReLU(negative_slope=leaky_slope)(model)
        
        return model
    return f


# Define the Relation Networks

def g_th(layers):
    def f(model):
        for n in range(len(layers)):
            model = layers[n](model)
        return model
    return f

def stack_layer(layers):
    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x
    return f

def g_theta(units=512, layers=4):
    r = []
    for k in range(layers):
        r.append(Dense(units))
        r.append(LeakyReLU(negative_slope=leaky_slope))
    return g_th(r)

def get_MLP():
    return g_th()


# Define the main RN
    
def RelationNetworks(objects):
    g_t = g_theta()
    relations = compute_relations(objects)
    print('No of Relations:', len(relations))
    #print(relations)
    
    g_all = []
    
    for i, r in enumerate(relations):
        g_all.append(g_t(r))
        
    combined_relation = Average()(g_all) # works best when combining relations
    
    f_out = f_theta()(combined_relation)
    return f_out

def build_tag(conv):
    d = K.int_shape(conv)[2]
    tag = np.zeros((d,d,2))
    
    for i in range(d):
        for j in range(d):
            tag[i,j,0] = float(int(i%d))/(d-1)*2-1
            tag[i,j,1] = float(int(j%d))/(d-1)*2-1
            
    tag = K.variable(tag)
    tag = K.expand_dims(tag, axis=0)
    batch_size = K.shape(conv)[0]
    tag = K.tile(tag, [batch_size,1,1,1])
    
    return Input(tensor=tag)


input_img = Input((x_train.shape[1], x_train.shape[2], 1))  
img_after_conv = ConvolutionNetworks()(input_img) 

img_after_RN = RelationNetworks(img_after_conv)

img_out = Dense(1, activation='linear')(img_after_RN)

RN_model = Model(inputs=[input_img], outputs=img_out)

###################################################################################
############################# MODEL COMPILATION ###################################
###################################################################################

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

#compile model
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

RN_model.compile(optimizer=adam, loss='mse', metrics=['mae','mape'])

# Train Model
history = RN_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list)

# Save the trained Model weights
save_weights_name = f"/path/to/sample_data/leaky_upper_triang_RN_model_GASE.weights.h5"
RN_model.save_weights(save_weights_name)

# Save the model architecture
save_arch_name = f"/path/to/sample_data/leaky_upper_triang_RN_model_GASE_arch.json"
with open(save_arch_name,'w') as f:
    f.write(RN_model.to_json())

# Save the training loss
train_loss_GASE = history.history['loss']
np.save(f"/path/to/sample_data/train_loss_leaky_dilate_CNN_RN_GASE",train_loss_GASE)

# Print Results
y_train_pred = RN_model.predict(x_train)
y_train_flat = y_train.flatten()  # or y_test.ravel()
y_train_pred_flat = y_train_pred.flatten()  # or y_test_pred.ravel()

train_corr_coef, train_p = pearsonr(y_train_flat, y_train_pred_flat)

# Test Model
y_test_pred = RN_model.predict(x_test)

print(y_test.shape)
print('y_test:', y_test)

print(y_test_pred.shape)
print('y_test_pred:', y_test_pred)

y_test_flat = y_test.flatten()  # or y_test.ravel()
y_test_pred_flat = y_test_pred.flatten()  # or y_test_pred.ravel()

# Print Results
print(f'Train mae: {np.mean(np.abs(y_train - y_train_pred)):.2f}, Train sdae: {np.std(np.abs(y_train - y_train_pred)):.2f}, Train corr: {train_corr_coef:.2f}, Train p-value: {train_p:.2f}')

file_name = f"/path/to/sample_data/y_train_pred_sample_data.mat"
scipy.io.savemat(file_name, {'y_train_pred': y_train_pred})

file_name = f"/path/to/sample_data/y_test_pred_sample_data.mat"
scipy.io.savemat(file_name, {'y_test_pred': y_test_pred})





