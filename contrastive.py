# Own libraries
import import_data
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

# Defining GPU use
GPU1 = "3" 
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = GPU1
 
# Creating .txt
stdoutOrigin=sys.stdout 
labeled_samples = 15000
path = '/home/kfonseca_cps/Pruebas_nuevas_K/Results/Contrastive_'+str(labeled_samples)+'/'
sys.stdout = open(path+'Contrastive_'+str(labeled_samples)+'.txt', "w")

# Dataset hyperparameters
signal_len = 2048
input_shape = (signal_len, 1)  
#labeled_samples = 10000 
validation_samples = 50000
unlabeled_samples = 200000
num_classes = 3

# Contrastive hyperparameters
batch_size_c = 256
batch_size = 256 
temperature = 0.1 
contrastive_epochs = 100
patience_stop_c = 8
patience_reduce_c = 4
projection_layers = 2
learning_rate = 1e-3
n_batches = int(unlabeled_samples*0.004)
val_batches = int(validation_samples*0.004)

# Baseline model parameters
baseline_epochs = 100 
width = 128
dense_width = 128
filters_initial = [4,8,16]
num_cnn = 13
residual = 4
kernel_size = 16
patience_stop = 15
patience_reduce = 4

print('[Info]: Code Parameters ')
print('Contrastive learning')
print('Training signals = ', str(labeled_samples))
print('Contrastive batch = ', str(batch_size_c*2))
print('Model: filters_initial = ', str(filters_initial))
print('num_cnn = ', str(num_cnn))
print('residual = ', str(residual))
print('Width = ', str(width))
print('Batch size = ', str(batch_size))
print('Projection layers = ',str(projection_layers))
print('Learning rate = ',str(learning_rate))
print('GPU = ', str(GPU1))
print('Comment: Codigo ordenado 2 ')

iterable = np.array([5000,10000,15000,20000,25000,50000,100000,200000])

for ite in iterable:

  for initial_filter in filters_initial:
  
    unlabeled_samples = ite
    n_batches = int(unlabeled_samples*0.004)
    print(" ####################################### UNLABELED SAMPLES - "+str(unlabeled_samples)+" #################################################################")
    # Initialization of variables
    
    fold_no = 1
    Acc_pre = []
    Loss_pre = []
    F1_pre = []
    Acc = []
    Loss = []
    F1_e = []
    train_c = []
    Acc_Val_per_fold = []
    F1_per_fold = []
    AUC_per_fold = []
    
    # Import training dataset 
    x_train, y_train = import_data.load_data_train('simple', signal_len, labeled_samples)
    x_test, y_test = import_data.load_data_test(signal_len)
    y_test = to_categorical(y_test, 3)
    # Kfolds training
    inputs = x_train
    targets = y_train
    num_folds = 10
    skf = StratifiedKFold(n_splits=num_folds, random_state=1,shuffle=True)
    print('Batch size = '+str(batch_size))
  
    print('[Info]: Hyperparameters pre-training ')
    print('Number of batches = '+str(n_batches))   
    
    x_train_c, _ = import_data.load_data_train('double',signal_len*2)
    train_c, val_c = import_data.preprocess_data_time_division(x_train_c,batch_size_c,n_batches,val_batches)
    del(x_train_c)
    
    print('Number of pre_train samples: ',train_c.shape[0]/2)
    print('Number of validation samples: ',val_c.shape[0]/2)
    
    train_c = tf.data.Dataset.from_tensor_slices(train_c).batch(2*batch_size_c,drop_remainder=True) 
    val_c = tf.data.Dataset.from_tensor_slices(val_c).batch(2*batch_size_c,drop_remainder=True)
    
    for train, test in skf.split(inputs, targets):
            
            callbacks_list = []
            acc = 0
            callbacks_list_c = []
            pre_model = []
            
            print('\n[Info]: Traning for the fold ', fold_no)
            
            x_train_fold  = inputs[train]
            y_train_fold = targets[train]
            x_val_fold = inputs[test] 
            y_val_fold = targets[test]
            
            y_train_fold = to_categorical(y_train_fold, 3)
            y_val_fold = to_categorical(y_val_fold, 3)
            
            train_dataset = (tf.data.Dataset.from_tensor_slices((x_train_fold,y_train_fold))
                        .shuffle(buffer_size=10*batch_size, seed= 0)
                        .batch(batch_size, drop_remainder = True))
            
            val_dataset = (tf.data.Dataset.from_tensor_slices((x_val_fold,y_val_fold))
                        .batch(batch_size, drop_remainder = True)
                        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
                        
            print('Train samples = ',x_train_fold.shape[0])
            print('Validation samples = ',x_val_fold.shape[0])         
               
            # Unlabeled llabels array assign 
            y = np.zeros(n_batches)
            y = tf.data.Dataset.from_tensor_slices(y).batch(1,drop_remainder=True) 
            pre_dataset = tf.data.Dataset.zip((train_c, y))
            y = np.zeros(val_batches)
            y = tf.data.Dataset.from_tensor_slices(y).batch(1,drop_remainder=True) 
            pre_dataset_val = tf.data.Dataset.zip((val_c, y)) 
            
            
            # Pre Training model
            print('\n[Info]: Pre-Training stage')
            while acc<0.81:
            
              del(callbacks_list_c)  # In case that pre-training accuracy doesn't pass
              del(pre_model)         # In case that pre-training accuracy doesn't pass
                      
              pre_model = import_functions.Contrastive_model_v2(initial_filter,num_cnn,residual,input_shape,width,projection_layers,'Pre_model')
              
              callbacks_list_c = import_functions.pre_callbacks(patience_stop_c,patience_reduce_c,path+'weights_best_contrastive_'+str(initial_filter)+'_'+str(ite)+'.h5')
              
              pretraining_history = pre_model.fit(
              pre_dataset, epochs=contrastive_epochs,verbose = 2,callbacks = callbacks_list_c, validation_data = pre_dataset_val) 
              
              acc = max(pretraining_history.history["val_simclr_accuracy"])
              print("Maximal validation accuracy: {:.2f}%".format(max(pretraining_history.history["val_simclr_accuracy"]) * 100))
                
            #pre_model.load_weights(model_path)
            post_model = import_functions.get_encoder(initial_filter,num_cnn,residual,input_shape,'Post_model')
            post_model.set_weights(pre_model.get_layer(pre_model.layers[0].name).get_weights())
            
            #post_model.trainable=False # To frozen weigths
            
            
            # Training stage
            print('\n[Info]: Training')
            
            model = keras.Sequential(
              [
                keras.Input(shape=(signal_len, 1)),
                post_model,
                layers.Dense(dense_width,kernel_initializer=tf.keras.initializers.HeNormal(seed=1)),
                layers.Dense(num_classes, activation = 'softmax'),
              ], 
              name="Fine_tuning_model",
            )
            
            model.summary()
            
            model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', tf.keras.metrics.AUC(),tfa.metrics.F1Score(num_classes)]
              )
              
            callbacks_list = import_functions.callbacks(patience_stop,patience_reduce, path+'weights_best_'+str(initial_filter)+'_'+str(ite)+'.h5')
              
            history = model.fit(
            train_dataset,epochs = baseline_epochs,callbacks = callbacks_list,verbose = 2, validation_data=val_dataset,)
            
            print('\n[Info]: Starting model evaluation: ')
            (loss, acc_full, AUC, F1) = model.evaluate(x_test,y_test, verbose=2)
            Acc_Val_per_fold.append(acc_full)
            F1_per_fold.append(np.mean(F1))
            AUC_per_fold.append(AUC)
            
            print('------------------------------------------------------------------------ ')
            print('\n[Info]: Updating Results')
            print(f'[Info]: Val_accu = {acc_full:.4} %')
            print('\n[Info]: F1 per fold = '+str(F1_per_fold))
            print('\n[Info]: Standard desviation F1= '+str(np.std(F1_per_fold)))
            print('\n[Info]: Mean F1= '+str(np.mean(F1_per_fold)))
            print('\n[Info]: AUC per fold = '+str(AUC_per_fold))
            print('\n[Info]: Standard desviation AUC= '+str(np.std(AUC_per_fold)))
            print('\n[Info]: Mean AUC= '+str(np.mean(AUC_per_fold)))
            
            
            fold_no = fold_no + 1
            del(model)
            del(pre_model)
            del(callbacks_list)
            del(post_model)
            del(callbacks_list_c)
            
    with open(path + '/Final_results.txt', 'a') as f:
      f.write('\n\n[Info]: Filter size = '+str(initial_filter))
      f.write('\n[Info]: Unlabeled data = '+str(ite))
      f.write('\n[Info]: F1 per fold = '+str(F1_per_fold))
      f.write('\n[Info]: Standard desviation F1= '+str(np.std(F1_per_fold)))
      f.write('\n[Info]: Mean F1= '+str(np.mean(F1_per_fold)))
      f.write('\n[Info]: AUC per fold = '+str(AUC_per_fold))
      f.write('\n[Info]: Standard desviation AUC= '+str(np.std(AUC_per_fold)))
      f.write('\n[Info]: Mean AUC= '+str(np.mean(AUC_per_fold)))
            
            
            
          
sys.stdout.close()
sys.stdout=stdoutOrigin