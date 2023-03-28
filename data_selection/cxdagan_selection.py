import numpy as np
import h5py
import time


import os
import sys

#%%

# OTHER COMMENTED LOCATIONS
#dataset = h5py.File('C:/Users/aleja/Documents/training_set_500.h5','r')
#dataset = h5py.File('C:/Users/USUARIO/Documents/Python Scripts/icentia_pkl_pruebas/training_set_500.h5','r')
#dataset = h5py.File('/home/kfonseca_cps/datasets_3/train_025_3.h5','r')
#dataset = h5py.File('/home/kfonseca_cps/dataset_doble/datasetd_05_2','r')
#test_set = h5py.File("/home/kfonseca_cps/Pruebas_Clustering/test_with_labels_200.h5",'r')
#test_set = h5py.File('/home/kfonseca_cps/datasets_3/test_3', 'r')
#training_set = h5py.File("/home/kfonseca_cps/Pruebas_Clustering/training_set_500.h5",'r') 

# PARAMETERS

GPU1 = "7"
os.environ["CUDA_DEVICE_ORDER"]    ="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = GPU1

stdoutOrigin=sys.stdout 
#path = '/home/kfonseca_cps/Pruebas_Clustering/Results/tsne_datasetd_05_2/raw2baselinefv2tsne/'
path = '/home/kfonseca_cps/Pruebas_Clustering/Results/test_cxdagan_selection/'
#stage = 'training/' #TRAINING OR PRETRAINING
stage = 'pretraining/' #TRAINING OR PRETRAINING
sys.stdout = open(path+stage+"logg_selection.txt", "w")

if stage == 'training/':
    path_source_data = '/home/kfonseca_cps/datasets_3/train_025_3.h5'
    print('TRAINING STAGE')
    #dim=2049
elif stage == 'pretraining/':
    path_source_data = '/home/kfonseca_cps/dataset_doble/datasetd_05_2'
    print('PRETRAINING STAGE')
    #dim=4097

dim=2049

#%% READ TARGET DATA
dataset = h5py.File('/home/kfonseca_cps/datasets_3/test_3', 'r')
#path = 'C:/Users/USUARIO/Documents/Python Scripts/selection_algorithm/'
#dataset = h5py.File(path+'test_with_labels_200.h5','r')

t = time.time()
print("READING DATA")
data_target = dataset.get('data')
data_target = np.array(data_target)
labels_target = dataset.get('labels')
labels_target = np.array(labels_target)
dataset.close()
print("TARGET DATA SHAPE: "+str(data_target.shape))
print("READ TARGET DATA ELAPSED TIME: "+str(time.time() - t))


#%% 
# CREATE AN ARRAY FOR EACH CLASS

data = data_target
labels = labels_target

classes = np.unique(labels).shape[0]
data_c=[] # data points for class n
indices_c=[] # indices for data points
for i in range(classes):
    data_c.append(data[np.where(labels==i)[0],:,0])
    indices_c.append(np.where(labels==i)[0])
    
data_target_c = data_c
indices_target_c = indices_c
    
#%% STANDARDIZED DATA (MEAN=0) AND EIGENVECTORS
# with std=0 the eigvectors were always the same direction

data = data_target_c

t = time.time()
eigval_c=[]
eigvec_c=[]
for i in range(classes):
    
    xy_standard = (data[i] - np.mean(data[i], axis=0))
    
    data[i] = xy_standard # overwrite data
    
    # COVARIANCE MATRIX
    covmat=np.cov(xy_standard,rowvar=False)
    
    # EIGENVECTORS AND EIGENVALUES
    eigval,eigvec=np.linalg.eig(covmat)
    eigvec=eigvec.transpose()
    
    eigval_c.append(eigval)
    eigvec_c.append(eigvec)
    print("EIGENVECTOR SHAPE: "+str(eigvec.shape))

data_target_c = data # overwrite in standardized data
eigval_target_c = eigval_c
eigvec_target_c = eigvec_c

print("FIND EIGENVECTORS ELAPSED TIME: "+str(time.time() - t))


#%% READ SOURCE DATA
dataset = h5py.File(path_source_data, 'r') 
#path = 'C:/Users/USUARIO/Documents/Python Scripts/selection_algorithm/'
#dataset = h5py.File(path+'test_with_labels_200.h5','r')

t = time.time()
print("READING DATA")
data_source = dataset.get('data')
data_source = np.array(data_source)
labels_source = dataset.get('labels')
labels_source = np.array(labels_source)
dataset.close()
print("SOURCE DATA SHAPE: "+str(data_source.shape))
print("READ SOURCE DATA ELAPSED TIME: "+str(time.time() - t))

if stage=='pretraining/':
    labels_source[labels_source==2]=1 # merge arrhythmias into 1 label
    labels_source[labels_source==3]=2 # relabel 4th label to have only 3 labels
    data_source = data_source[:,0:dim] # just selecting first 2049 samples
    '''
    # RANDOMLY SELECTING THE FIRST OR SECOND HALF OF EACH SIGNAL
    print("signals shape: " + str(signals.shape))
    signal_len = data.shape[1] - 1
    signals = signals[:,0:signal_len] # ignoring last sample to make it even
    nsignals = signals.shape[0]
    print("signals shape: " + str(signals.shape))

    signals1, signals2 = np.split(signals, 2, axis=1) # split signals in 2

    # reshape it so i can concatenate it with 2 separate parts (first half and second half)
    signals1 = np.reshape(signals1, (signals1.shape[0],signals1.shape[1],1))
    signals2 = np.reshape(signals2, (signals2.shape[0],signals2.shape[1],1))
    new_signals = np.concatenate((signals1, signals2),axis=2)

    indices2=np.random.randint(2, size=signals.shape[0]).astype('int64') # an array with length nsignals with values 1 or 0

    # create new array with the first or second half of the signal (random)
    random_new_signals = np.zeros(signals1[:,:,0].shape) #empty array size=(nsignals,signal_len/2)
    for i in range(signals.shape[0]):
        random_new_signals[i,:] = new_signals[i,:,indices2[i]]
    '''

#%% 
# CREATE AN ARRAY FOR EACH CLASS

#nsignals = 300#data_source.shape[0]
data = data_source
labels = labels_source

classes, class_counts = np.unique(labels, return_counts=True)
classes=classes.shape[0]
print("CLASSES: "+str(classes))
print("CLASS COUNTS: "+str(class_counts))

#class_counts = [250,220,270] # TO MODIFY COUNTS USED OR NOT
print("CLASS COUNTS USED: "+str(class_counts))

data_c=[] # data points for class n
indices_c=[] # indices for data points
labels_c=[] # labels for data points
for i in range(classes):
    data_c.append(data[np.where(labels==i)[0][0:class_counts[i]],:,0])
    indices_c.append(np.where(labels==i)[0][0:class_counts[i]])
    labels_c.append(labels[np.where(labels==i)[0][0:class_counts[i]],0])
    
data_source_c = data_c
indices_source_c = indices_c
labels_source_c = labels_c

#%% 
# PROJECTION ON EIGENVECTORS

eig_iter = 200 # number of vectors for base
#data_iter = nsignals # number of signals to be projected

t = time.time()
eigval_c = eigval_target_c
eigvec_c = eigvec_target_c
data_c = data_source_c
data_iter=[]
for i in range(classes):
    data_iter.append(data_c[i].shape[0])
print("data_iter: ")
print(data_iter)
data_projected_c=[]
for l in range(classes):
    projected_temp = []
    # PROJECTION OF CLASS L ONTO EACH OTHER CLASS
    for k in range(classes):
        vp=np.zeros((data_iter[l],dim)) # vectors projected onto base
        # PROJECTION OF VECTOR ONTO BASIS
        for j in range(data_iter[l]):
            a=data_c[l][j]#a=data_c_i # # array chosen to be projected onto the base
            
            for i in range(eig_iter): 
                b=eigvec_c[k][i].reshape(dim,1) # base vector
                
                # projection formula
                vp_temp = np.dot(b,(np.dot(a,b) / (np.linalg.norm(b)**2)).transpose())
                vp[j] = vp[j] + vp_temp
        projected_temp.append(vp)
    
    data_projected_c.append(projected_temp)
    
data_projected_source_c = data_projected_c
print("PROJECTION ON EIGENVECTORS ELAPSED TIME: "+str(time.time() - t))    

'''
print('Crear h5 file')
hf = h5py.File(path+stage+'data_projected_source_c.h5', 'w')

for i in range(classes):
    for j in range(classes):
        
        hf.create_dataset('data_projected_source_c_'+str(i)+'_'+str(j), data=data_projected_source_c[i][j], compression='gzip')

hf.close()
'''

#%%
# ERRORS 

data_c = data_source_c
data_projected_c = data_projected_source_c

t = time.time()
error_n=[]
for l in range(classes):
    errors_temp=[]
    for k in range(classes):
        errors_temp.append(np.linalg.norm(data_c[l] - data_projected_c[l][k],axis=1))
    error_n.append(errors_temp)
    
error_c = error_n
print("FIND ERRORS ELAPSED TIME: "+str(time.time() - t)) 


        
#%% SORTING ERRORS



error_c = error_c
indices_c = indices_source_c
labels_c = labels_source_c

t = time.time()
error_sorted_c = []
error_indices_sorted_c = []
indices_sorted_c = []
#labels_sorted_c = []
for l in range(classes):
    error_sorted_c_temp = []
    error_indices_sorted_c_temp = []
    indices_sorted_c_temp = []
    #labels_sorted_c_temp = []
    for k in range(classes):
        if k==l:
            # INTRA-CLASS
            error_sorted_c_temp.append(np.sort(error_c[l][k])) # acscending order
            error_indices_sorted_c_temp.append(np.argsort(error_c[l][k]))
            indices_sorted_c_temp.append(indices_c[l][error_indices_sorted_c_temp[k]])
            #labels_sorted_c_temp.append(labels_c[l][error_indices_sorted_c_temp[k]])
            
        else:
            # INTER-CLASS
            error_sorted_c_temp.append(np.sort(error_c[l][k])[::-1]) # descending order
            error_indices_sorted_c_temp.append(np.argsort(error_c[l][k])[::-1])
            indices_sorted_c_temp.append(indices_c[l][error_indices_sorted_c_temp[k]])
            #labels_sorted_c_temp.append(labels_c[l][error_indices_sorted_c_temp[k]])
            
        
    error_sorted_c.append(error_sorted_c_temp)
    error_indices_sorted_c.append(error_indices_sorted_c_temp)
    indices_sorted_c.append(indices_sorted_c_temp)
    #labels_sorted_c.append(labels_sorted_c_temp)

error_sorted_c = error_sorted_c
error_indices_sorted_c = error_indices_sorted_c# i don't think this one is needed
indices_source_sorted_c = indices_sorted_c
#labels_source_sorted_c = labels_sorted_c# also not this one
print("SORT ERRORS ELAPSED TIME: "+str(time.time() - t)) 

#%% TO VERIFY SORTING
l=1
k=2
temp = error_c[l][k]
if l==k:
    error_sorted = np.sort(temp) 
    error_indices_sorted = np.argsort(temp)
else:
    error_sorted = np.sort(temp)[::-1] 
    error_indices_sorted = np.argsort(temp)[::-1]

# to verify
#data_sorted = data_source_c[0][error_indices_sorted]
#data_projected_sorted = data_projected_c[0][0][error_indices_sorted]
#print(np.linalg.norm(data_sorted - data_projected_sorted,axis=1))

# to verify
indices_sorted = indices_c[l][error_indices_sorted]
data_sorted = data_source[indices_sorted,:,0]
data_projected_sorted = data_projected_c[l][k][error_indices_sorted]
print("VERIFYING: l="+str(l)+" k= "+str(k))
print(np.linalg.norm(data_sorted - data_projected_sorted,axis=1)) # error_sorted_c[l][k]
print("error_sorted_c[l][k]: ")
print(error_sorted_c[l][k])
print("\n\n\n")
print(indices_sorted) # indices_source_sorted_c[l][k]
print("indices_source_sorted_c[l][k]: ")
print(indices_source_sorted_c[l][k])


#%% SAVING VARIABLES

print('Crear h5 file')
hf = h5py.File(path +stage+ 'error_sorted_c.h5', 'w')

for i in range(classes):
    for j in range(classes):
        
        hf.create_dataset('error_sorted_c_'+str(i)+'_'+str(j), data=error_sorted_c[i][j], compression='gzip')

hf.close()

print('Crear h5 file')
hf = h5py.File(path +stage+ 'indices_source_sorted_c.h5', 'w')

for i in range(classes):
    for j in range(classes):
        
        hf.create_dataset('indices_source_sorted_c_'+str(i)+'_'+str(j), data=indices_source_sorted_c[i][j], compression='gzip')

hf.close()
'''
print('Crear h5 file')
hf = h5py.File(path + 'labels_source_sorted_c.h5', 'w')

for i in range(classes):
    for j in range(classes):
        
        hf.create_dataset('labels_source_sorted_c_'+str(i)+'_'+str(j), data=labels_source_sorted_c[i][j], compression='gzip')

hf.close()
'''
#%% SEE WHAT INDICES ARE IN THE TOP
# TURNED OUT TO BE NOT VERY USEFUL
'''
indices_sorted_c = indices_source_sorted_c

t = time.time()
nsignals = data_source.shape[0]
ntop = 200
indices_count_inter = np.zeros(nsignals)
indices_count_intra = np.zeros(nsignals)
for i in range(nsignals):
    for l in range(classes):
        for k in range(classes):
            if i in indices_sorted_c[l][k][0:ntop]: #searches the first "ntop" values
                if l==k:
                    indices_count_intra[i] += 1
                else:
                    indices_count_inter[i] += 1
                


indices_count_total = indices_count_inter + indices_count_intra

indices_count_total_sort = np.sort(indices_count_total)
indices_count_total_argsort = np.argsort(indices_count_total)
indices_count_total_argsort_labels = labels_source[indices_count_total_argsort]
print("COUNT AND SORT INDICES ELAPSED TIME: "+str(time.time() - t)) 

ntemp=100#data.source.shape[0]
print("indices_count_total_sort: ")
print(indices_count_total_sort[0:ntemp])
print("indices_count_total_argsort: ")
print(indices_count_total_argsort[0:ntemp])
print("indices_count_total_argsort_labels: ")
print(indices_count_total_argsort_labels[0:ntemp])




print('Crear h5 file')
hf = h5py.File(path + 'indices_count_sorted.h5', 'w')
hf.create_dataset('indices_count_inter', data=indices_count_inter, compression='gzip')
hf.create_dataset('indices_count_intra', data=indices_count_intra, compression='gzip')
hf.create_dataset('indices_count_total', data=indices_count_total, compression='gzip')
hf.create_dataset('indices_count_total_sort', data=indices_count_total_sort, compression='gzip')
hf.create_dataset('indices_count_total_argsort', data=indices_count_total_argsort, compression='gzip')
hf.create_dataset('indices_count_total_argsort_labels', data=indices_count_total_argsort_labels, compression='gzip')
hf.close()

'''



sys.stdout.close()
sys.stdout=stdoutOrigin




















