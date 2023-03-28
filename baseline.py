# -*- coding: utf-8 -*-

# Own libraries
import import_data
import import_functions

import numpy as np 
import numpy.matlib
import os 
import matplotlib.pyplot as plt 

# Training libraries
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers 
from tensorflow.keras import layers
from keras import regularizers
from keras.layers import LayerNormalization
from sklearn.model_selection import StratifiedKFold

#  Data libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.data import AUTOTUNE
import h5py
import sys 

stdoutOrigin=sys.stdout 


labeled_samples = 15000
path = '/home/kfonseca_cps/Pruebas_nuevas_K/Results/Baseline_'+str(labeled_samples)+'/'

sys.stdout = open(path+'Base2_'+str(labeled_samples)+'.txt', "w") #File name

GPU1 = "4"
os.environ["CUDA_DEVICE_ORDER"]    ="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = GPU1

# Hyperparameters
signal_len = 2048
num_classes = 3
batch_size = 256 

# Baseline model parameters
baseline_epochs = 100 
dense_width = 128
width =  128
filters_initial = [4]
num_cnn = 13
residual = 4
kernel_size = 16
patience_stop = 15
patience_reduce = 4


print('\n[Info]: Import data')

x_train_f, y_train_f = import_data.load_data_train('simple', signal_len, labeled_samples)
print('\n[Info]: Simple train data correctly imported')

x_test, y_test = import_data.load_data_test(signal_len)
print('\n[Info]: Test data correctly imported')

y_test = to_categorical(y_test, 3)

#test_dataset = (tf.data.Dataset.from_tensor_slices((x_test,y_test))
#          .batch(batch_size, drop_remainder = True)
#          .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))

print('\n[Info]: Size for x_train_f = '+str(x_train_f.shape))

## K-folds settings

num_folds = 10
skf = StratifiedKFold(n_splits=num_folds, random_state=0, shuffle=True)

## Callbacks



inputs = x_train_f
targets = y_train_f

for initial_filter in filters_initial:

  fold_no = 1
  Acc_Val_per_fold = []
  F1_per_fold = []
  AUC_per_fold = []

  for train, test in skf.split(inputs, targets):
  
    print('\n[Info]: Traning for the fold ', fold_no)
    
    callbacks_list = import_functions.callbacks(patience_stop,patience_reduce, path+'weights_best_'+str(initial_filter)+'.h5')
    
    ##Baseline model
    
    print('\n[Info]: Starting training: ')
    
    x_train_fold  = inputs[train] 
    y_train_fold = targets[train]
    print('Tamaño de x_train_fold = '+str(x_train_fold.shape))
    x_val_fold = inputs[test] 
    y_val_fold = targets[test]
    print('Tamaño de validacion = '+str(x_val_fold.shape))
  
    classes = tf.math.bincount(y_train_fold)
    classes = np.array(classes)
    print('[Info]: x_train_fold:')
    print('Number of train samples: ',y_train_fold.shape[0])
    print('Distribution of classes: ',classes)
    
    classes = tf.math.bincount(y_val_fold)
    classes = np.array(classes)
    print('[Info]: Validation:')
    print('Number of train samples: ',y_val_fold.shape[0])
    print('Distribution of classes: ',classes)
    
    y_train_fold = to_categorical(y_train_fold, 3)
    y_val_fold = to_categorical(y_val_fold, 3)
    
    labeled_train_dataset = (tf.data.Dataset.from_tensor_slices((x_train_fold,y_train_fold))
                        .shuffle(buffer_size=10*batch_size, seed= 0)
                        .batch(batch_size, drop_remainder = True))
                        
    val_dataset = (tf.data.Dataset.from_tensor_slices((x_val_fold,y_val_fold))
              .batch(batch_size, drop_remainder = True)
              .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
    
    model = keras.Sequential(
        [
            keras.Input(shape=(signal_len, 1)),
            import_functions.get_encoder(initial_filter,num_cnn,residual,(signal_len, 1),'ecg_model'),
            layers.Dense(dense_width,kernel_initializer=tf.keras.initializers.HeNormal(seed=1)),
            layers.Dense(num_classes, activation = 'softmax'), #Activation
        ],
        name="model",
    )
    
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes)],
        )
    num_layers = len(model.layers)-1   
    
    if fold_no == 1:
      model.summary()
      print(labeled_train_dataset)
      print(val_dataset)
      
    history = model.fit(labeled_train_dataset, 
                                 epochs = baseline_epochs, 
                                 batch_size = batch_size,
                                 callbacks = callbacks_list,
                                 validation_data = val_dataset,
                                 verbose = 2) 
    
    print('\n[Info]: Starting model evaluation: ')
    (loss, acc_full, AUC, F1) = model.evaluate(x_test,y_test, verbose=2)
    print(f'[Info]: Val_accu = {acc_full:.4} %')
    Acc_Val_per_fold.append(acc_full)
    F1_per_fold.append(np.mean(F1))
    AUC_per_fold.append(AUC)
    
    print('------------------------------------------------------------------------ ')
    print('\n[Info]: Updating Results')
    print('\n[Info]: F1 per fold = '+str(F1_per_fold))
    print('\n[Info]: Standard desviation F1= '+str(np.std(F1_per_fold)))
    print('\n[Info]: Mean F1= '+str(np.mean(F1_per_fold)))
    print('\n[Info]: AUC per fold = '+str(AUC_per_fold))
    print('\n[Info]: Standard desviation AUC= '+str(np.std(AUC_per_fold)))
    print('\n[Info]: Mean AUC= '+str(np.mean(AUC_per_fold)))
    fold_no = fold_no + 1
  
    del(model)
    
  with open(path + '/Final_results.txt', 'a') as f:
    f.write('\n\n[Info]: Filter size = '+str(initial_filter))
    f.write('\n[Info]: F1 per fold = '+str(F1_per_fold))
    f.write('\n[Info]: Standard desviation F1= '+str(np.std(F1_per_fold)))
    f.write('\n[Info]: Mean F1= '+str(np.mean(F1_per_fold)))
    f.write('\n[Info]: AUC per fold = '+str(AUC_per_fold))
    f.write('\n[Info]: Standard desviation AUC= '+str(np.std(AUC_per_fold)))
    f.write('\n[Info]: Mean AUC= '+str(np.mean(AUC_per_fold)))

sys.stdout.close()
sys.stdout=stdoutOrigin