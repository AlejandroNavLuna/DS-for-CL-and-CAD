import h5py
import numpy as np
import scipy.stats as scst
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import time

import os
import sys

#%%

GPU1 = "5"
os.environ["CUDA_DEVICE_ORDER"]    ="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = GPU1
stdoutOrigin=sys.stdout 
path_read = "/home/kfonseca_cps/Pruebas_Clustering/Results/tsne_train_025_3/raw2fv2tsne/"
path = "/home/kfonseca_cps/Pruebas_Clustering/Results/tsne_train_025_3/raw2fv2tsne/generated/"
sys.stdout = open(path + "generate_logg.txt", "w")

#dataset = h5py.File('C:/Users/aleja/Documents/training_set_500.h5','r')
#dataset = h5py.File('C:/Users/USUARIO/Documents/Python Scripts/icentia_pkl_pruebas/training_set_500.h5','r')
#  /home/kfonseca_cps/dataset_doble/datasetd_05_2
dataset = h5py.File('/home/kfonseca_cps/dataset_doble/datasetd_05_2','r')
#'D:/main code other datasets/datasets_3/train_025_3.h5', 'r') 


print("READING DATA")
data = dataset.get('data')
data = np.array(data)
labels = dataset.get('labels')
labels = np.array(labels)
patients = dataset.get('patients')
patients = np.array(patients)
dataset.close()

# NUMBER OF SAMPLES TAKEN
nsignals_start = 0#2000#data.shape[0]
nsignals_finish = 30000#data.shape[0]#4000
signals = data[nsignals_start:nsignals_finish,:,0]
signal_labels = labels[nsignals_start:nsignals_finish,0]
patient_labels = labels[nsignals_start:nsignals_finish,0]

print("NUMBER OF SIGNALS TAKEN: " + str(signals.shape[0]))

#%%###################### READ AND CREATE SET ARRAYS ##############

nsignals = signals.shape[0]
iterations = 3000
tsne_results_sets = h5py.File(path_read + 'clustering_sets/tsne_results_' + str(nsignals) + '_' + str(iterations) + '_iter.h5','r')

tsne_results = tsne_results_sets.get('tsne_results')
#set_1 = tsne_results_sets.get('set_1')
#set_1_indices = tsne_results_sets.get('set_1_indices')
#set_2 = tsne_results_sets.get('set_2')
#set_2_indices = tsne_results_sets.get('set_2_indices')
#set_3 = tsne_results_sets.get('set_3')
#set_3_indices = tsne_results_sets.get('set_3_indices')
#set_4 = tsne_results_sets.get('set_4')
#set_4_indices = tsne_results_sets.get('set_4_indices')
print("tsne_results shape: " + str(tsne_results.shape))





'''

#%%############################ t-SNE ############################

iterations = 3000

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=iterations)
tsne_results = tsne.fit_transform(fv)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
'''


#%%############################ TOTAL CLUSTER IMAGES ###################



time_start = time.time()

fig, ax = plt.subplots(figsize=(16,10))  

colors = ['blue','green','orange']

scatter = ax.scatter(
    x=tsne_results[:,0],
    y=tsne_results[:,1],
    c=signal_labels.astype('int64'),
    alpha=0.3,
    cmap=mcolors.ListedColormap(colors)
    )

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")

#plt.savefig('/home/kfonseca_cps/Pruebas_Clustering/Results/tsne_train_025_3/raw2fv2tsne/total_cluster.png')
plt.savefig(path + "images/total_cluster_" + str(signals.shape[0]) + "_" + str(iterations) + "_iter.png")

#plt.show()


print('time elapsed for plotting figure: {} seconds'.format(time.time()-time_start))

# TOTAL CLUSTER IMAGE OF EACH CLASS

for i in range(np.unique(signal_labels).shape[0]):
    
    # SELECT INDICES THAT BELONG TO THE CLASS
    class_label = np.unique(signal_labels)[i]
    class_indices = np.where(signal_labels == class_label)
    tsne_results_class=tsne_results[class_indices]
    
    # FIGURE
    
    fig, ax = plt.subplots(figsize=(16,10))  
    
    colors = ['blue','green','orange']
    
    scatter = ax.scatter(
        x=tsne_results_class[:,0],
        y=tsne_results_class[:,1],
        c=signal_labels[class_indices].astype('int64'),
        alpha=0.3,
        cmap=mcolors.ListedColormap(colors[class_label.astype('int64')])
        )
    
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    
    plt.savefig(path + "images/total_cluster_" + str(signals.shape[0]) + "_iter_" + str(iterations) + "_class_" + str(int(class_label)) + ".png")
    
    #plt.show()

'''
#%%######################################## CONDITIONS TO CREATE SETS ###################################

set_1_indices =np.array([])
set_1=np.array([])
set_2_indices =np.array([])
set_2=np.array([])
set_3_indices =np.array([])
set_3=np.array([])
set_4_indices =np.array([])
set_4=np.array([])

# GENERATE NEW ARRAYS BASED ON THE CONDITION
for i in range(tsne_results.shape[0]):
    if tsne_results[i,0]>0 and tsne_results[i,1]>0: # CONDITION
    
        set_1=np.append(set_1, tsne_results[i,:])
        set_1_indices=np.append(set_1_indices, i)
        
    elif tsne_results[i,0]<0 and tsne_results[i,1]>0: # CONDITION
    
        set_2=np.append(set_2, tsne_results[i,:])
        set_2_indices=np.append(set_2_indices, i)
        
    elif tsne_results[i,0]<0 and tsne_results[i,1]<0: # CONDITION
    
        set_3=np.append(set_3, tsne_results[i,:])
        set_3_indices=np.append(set_3_indices, i)
        
    elif tsne_results[i,0]>0 and tsne_results[i,1]<0: # CONDITION
    
        set_4=np.append(set_4, tsne_results[i,:])
        set_4_indices=np.append(set_4_indices, i)
        
        
# RESHAPE BECAUSE THE FOR APPEND THE x AND y VALUES IN SERIES
# SO ITS TO MAKE IT LIKE THIS: [[x1,y1],[x2,y2],...]
set_1 = np.reshape(set_1, (set_1_indices.shape[0], -1))
set_2 = np.reshape(set_2, (set_2_indices.shape[0], -1))
set_3 = np.reshape(set_3, (set_3_indices.shape[0], -1))
set_4 = np.reshape(set_4, (set_4_indices.shape[0], -1))

#%%######################################## SET FIGURES AND H5 FILES ###################################

# lists with set info
sets = [set_1,set_2,set_3,set_4]
sets_indices = [set_1_indices, set_2_indices, set_3_indices, set_4_indices,]

for i in range(len(sets)):
  
  signals_set = data[sets_indices[i].astype('int64')]
  signal_labels_set = labels[sets_indices[i].astype('int64')]
  patient_labels_set = labels[sets_indices[i].astype('int64')]
  
  # FIGURES
  
  fig, ax = plt.subplots(figsize=(16,10))  
  scatter = ax.scatter(
      x=sets[i][:,0],
      y=sets[i][:,1],
      c=labels[sets_indices[i].astype('int64')].astype('int64'),
      alpha=0.3,
      cmap=mcolors.ListedColormap(colors)
      )
  # produce a legend with the unique colors from the scatter
  legend1 = ax.legend(*scatter.legend_elements(),
                      loc="lower left", title="Classes")
  plt.savefig(path + 'images/set_' + str(i+1) + '_' + str(signals_set.shape[0]) + '_' + str(iterations) + '_iter.png')
  #plt.show()
  
  # H5 FILES
    
  print('Crear set ' + str(i+1) + ' h5')
  print('Set size: ' + str(signals_set.shape))
  hf = h5py.File(path + 'clustering_sets/set_' + str(i+1) + '_' + str(signals_set.shape[0]) + '_' + str(iterations) + '_iter.h5', 'w')
  hf.create_dataset('data', data=signals_set, compression='gzip')
  hf.create_dataset('labels', data=signal_labels_set, compression='gzip')
  hf.create_dataset('patients', data=patient_labels_set, compression='gzip')
  hf.close()
#'''

print('Crear h5 file with tsne_results, sets, and sets with indices')
hf = h5py.File(path + 'clustering_sets/tsne_results_' + str(signals.shape[0]) + '_' + str(iterations) + '_iter.h5', 'w')
hf.create_dataset('tsne_results', data=tsne_results, compression='gzip')
hf.close()

#'''



sys.stdout.close()
sys.stdout=stdoutOrigin





















