from util import*

import numpy as np
from scipy.spatial.distance import pdist

def toDistance(R):
    '''
    This function takes a numpy array containing positions and returns it as distances.
    
    Parameters:
        -R:
            numpy array containing positions for every atom in every sample
            Dimensions: (n_samples,n_atoms,n_dimensions)
            
    Returns:
        -y:
            numpy array containing distances for every atom in every sample
            Dimensions: (n_samples,n_atoms*(n_atoms-1)/2)
    '''
    
    shape=R.shape
    try:
        dim=shape[2]
    except:
        print("toDistance: wrong dimensions")
        return
    if shape[1]<2:
        print("not enough atoms")
        return

    y=[]

    for i in range(len(R)): ##goes through samples
        y.append(pdist(R[i]))

    y=np.array(y)
    return y

def r_to_desc(data_set):
    '''
    Returns the position array as an array of desired description.
    This description is solely used for clusterisation.
    Default is inverse distances.
    
    Parameters:
        -R:
            numpy array containing positions for every atom in every sample
            Dimensions: (n_samples,n_atoms,n_dimensions)
                    
    Returns:
        numpy array containing inverse distances for every atom in every sample
        Dimensions: (n:samples,n_atoms*(n_atoms-1)/2)
    '''

    R=data_set['R']
    return 1. / toDistance(R)

def extract_E(data_set):

    E=data_set['E']
    return np.array(E)

def extract_R_concat(data_set):

    R=data_set['R']
    n_samples,n_atoms,n_dim=R.shape
    R=np.reshape(R,(n_samples,n_atoms*n_dim))
    return np.array(R)

def extract_F_concat(data_set):

    F=data_set['F']
    n_samples,n_atoms,n_dim=F.shape
    F=np.reshape(F,(n_samples,n_atoms*n_dim))
    return np.array(F)

def mean_squared_error_sample_wise(x,y):
    err=(np.array(x)-np.array(y))**2
    return err.mean(axis=1)

def lowest_variance_ind(a):
    var_sum=[]
    for s in a:
        b=a-s
        var_sum.append(np.sum(np.square(b)))
    return np.argmin(var_sum)

def within_cluster_lowest_variance(db,cl_ind,err):
    new_ind=[]
    var=db.vars[db.para['generate_training_data']['var_index']]
    for ind in cl_ind:
        R=var[ind]
        lv_ind=lowest_variance_ind(R)
        new_ind.append(ind[lv_ind])

    return new_ind

def within_cluster_highest_error(db,cl_ind,err):
    new_ind=[]
    for ind in cl_ind:
        cl_err=err[ind]
        argmax=np.argmax(cl_err)
        new_ind.append(ind[argmax])

    return new_ind

def within_high_error_cluster_lowest_variance(db,cl_ind,err):
    new_ind=[]
    var=db.vars[db.para['generate_training_data']['var_index']]
    N=db.para['generate_training_data']['n_he_clusters']

    #find highest error clusters
    mse=[np.mean(err[x]) for x in cl_ind]
    argsort=np.argsort(mse)
    new_cl_ind=np.array(cl_ind)[argsort[-N:]]

    for ind in new_cl_ind:
        R=var[ind]
        lv_ind=lowest_variance_ind(R)
        new_ind.append(ind[lv_ind])

    return new_ind



