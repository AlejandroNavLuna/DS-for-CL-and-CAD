# Own libraries
#import import_data
import import_functions

# Training libraries
import os 
import sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.layers import LayerNormalization

# for figures
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

import os
import sys

#%%

GPU1 = "5"
os.environ["CUDA_DEVICE_ORDER"]    ="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = GPU1
stdoutOrigin=sys.stdout 
path = '/home/kfonseca_cps/Pruebas_Clustering/'
sys.stdout = open(path + "logg_generation.txt", "w")

#dataset = h5py.File('C:/Users/aleja/Documents/training_set_500.h5','r')
#dataset = h5py.File('C:/Users/USUARIO/Documents/Python Scripts/icentia_pkl_pruebas/training_set_500.h5','r')
#dataset = h5py.File('/home/kfonseca_cps/datasets_3/train_025_3.h5','r')
dataset = h5py.File('/home/kfonseca_cps/dataset_doble/datasetd_05_2','r')
#'D:/main code other datasets/datasets_3/train_025_3.h5', 'r') 



print("READING DATA")
data = dataset.get('data')
data = np.array(data)
labels = dataset.get('labels')
labels = np.array(labels)
#patients = dataset.get('patients')
#patients = np.array(patients)
dataset.close()

print("SIGNALS SHAPE: " + str(data.shape))
print("LABELS SHAPE: " + str(labels.shape))
'''
# NUMBER OF SAMPLES TAKEN
nsignals_start = 0#2000
nsignals_finish = data.shape#10000
signals = data[nsignals_start:nsignals_finish,:,0]
signal_labels = labels[nsignals_start:nsignals_finish,0]
patient_labels = labels[nsignals_start:nsignals_finish,0]

print("NUMBER OF SIGNALS TAKEN: " + str(signals.shape[0]))
#'''

#  FIND SIZE OF EACH CLASS
indices_0 = np.where(labels==0)[0]
data_0 = data[indices_0]
labels_0 = labels[indices_0]
print("signals with label labels 0: " +str(data_0.shape[0]))
indices_1 = np.where(labels==1)[0]
data_1 = data[indices_1]
labels_1 = labels[indices_1]
print("signals with label labels 1: " +str(data_1.shape[0]))
indices_2 = np.where(labels==2)[0]
data_2 = data[indices_2]
labels_2 = labels[indices_2]
print("signals with label labels 2: " +str(data_2.shape[0]))
indices_3 = np.where(labels==3)[0]
data_3 = data[indices_3]
labels_3 = labels[indices_3]
print("signals with label labels 3: " +str(data_3.shape[0]))

nsignals=20000

# SELECT SIZE OF EACH CLASS SO THAT IT IS BALANCED
remainder=np.mod(nsignals - (data_1.shape[0] + data_2.shape[0]),2)
print("nsignals: " + str(nsignals))
print("remainder: " + str(remainder))
print("datashape1: " + str(data_1.shape[0]))
print("datashape2: " + str(data_2.shape[0]))
nsignals_class_0 = (nsignals - (data_1.shape[0] + data_2.shape[0]) - remainder)/2 + remainder
nsignals_class_3 = (nsignals - (data_1.shape[0] + data_2.shape[0]) - remainder)/2
'''
if remainder==1:
    nsignals_class_0 = (nsignals - (data_1.shape[0] + data_2.shape[0]) - remainder)/2 + 1
    nsignals_class_3 = (nsignals - (data_1.shape[0] + data_2.shape[0]) - remainder)/2
elif remainder==0:
    nsignals_class_0 = (nsignals - (data_1.shape[0] + data_2.shape[0]))/2
    nsignals_class_3 = (nsignals - (data_1.shape[0] + data_2.shape[0]))/2
#'''
    

print("nsignals_class_0: " + str(nsignals_class_0))
print("nsignals_class_3: " + str(nsignals_class_3))


#len_indices_0 = [len(a) for a in indices_0]
#print("size of indices_0: " + str(len_indices_0))
#print(str(indices_0[0]))
#print(str(indices_0[1][0:100]))
indices_random_0 = np.random.choice(indices_0, int(nsignals_class_0)) # randomly picks out x amount (nsignals_class) of indices from indices_0
print("indices_random_0: " + str(indices_random_0))
data_out_0 = data[indices_random_0]
labels_out_0 = labels[indices_random_0]
print("labels of indices_random_0: " + str(labels_out_0))


indices_random_3 = np.random.choice(indices_3, int(nsignals_class_3))
print("indices_random_3: " + str(indices_random_3))
data_out_3 = data[indices_random_3]
labels_out_3 = labels[indices_random_3]
print("labels of indices_random_3: " + str(labels_out_3))
#'''

print("size of each class array: " + str(data_out_0.shape))
print("size of each class array: " + str(data_1.shape))
print("size of each class array: " + str(data_2.shape))
print("size of each class array: " + str(data_out_3.shape))

# OUTPUT BALANCED DATASET
data_out = np.append(data_out_0, np.append(data_1, np.append(data_2, data_out_3, 0), 0), 0)
labels_out = np.append(labels_out_0, np.append(labels_1, np.append(labels_2, labels_out_3, 0), 0), 0)

print("SIZE DATA OUT: " + str(data_out.shape))
print("SIZE LABELS OUT: " + str(labels_out.shape)) 


# H5 FILE
print('Crear h5 file ')
hf = h5py.File(path + 'datasetd_05_2_'+str(int(nsignals))+'_balanced.h5', 'w')
hf.create_dataset('data', data=data_out, compression='gzip')
hf.create_dataset('labels', data=labels_out, compression='gzip')
hf.close()



sys.stdout.close()
sys.stdout=stdoutOrigin





















