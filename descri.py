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

def r_to_desc(self):
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
    dataset=self.dataset
    R=dataset['R']
    return 1. / toDistance(R)

def extract_E(self):
    dataset=self.dataset
    E=dataset['E']
    return np.array(E)

def extract_R_concat(self):
    dataset=self.dataset
    R=dataset['R']
    n_samples,n_atoms,n_dim=R.shape
    R=np.reshape(R,(n_samples,n_atoms*n_dim))
    return np.array(R)

def extract_F_concat(self):
    dataset=self.dataset
    F=dataset['F']
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

def weighted_distribution(N,weights):
    weights=np.array(weights)/np.sum(weights)
    a=(weights*N)
    b=a.astype(int)
    c=a-b
    s=np.sum(b)

    for i in range(N-s):
        arg=np.argmax(c)
        c[arg]=0
        b[arg]=b[arg]+1

    return b

def db_indices(self):
    return np.arange(len(self.dataset))

def within_cluster_weighted_err_N(db,cl_ind,err):
    new_ind=[]
    N=db.para['generate_training_data']['n_points']

    #find cluster errors and pops
    mse=np.array([np.mean(err[x]) for x in cl_ind])
    pop=np.array([len(x) for x in cl_ind])

    weights=(mse/np.sum(mse))*(pop/np.sum(pop))
    Ns=weighted_distribution(N,weights)

    for  i in range(len(cl_ind)):
        ind=np.array(cl_ind[i])
        cl_err=err[ind]
        ni=Ns[i]
        argmax=np.argsort(-cl_err)[:ni]
        new_ind.extend(ind[argmax])

    return new_ind

def MD17_extract_E(self):

    if getattr(self,'dataset_npz',None) is None:
        dataset=self.dataset
        E=[]
        l=len(dataset)    
        for i in range(l):
            E.append(dataset[i]['energy'].numpy())
            print(f"Exctracting db energies: {i/l*100:.0f}%",end='\r')

        print("")
    else:
        E=np.reshape(self.dataset_npz['E'],(len(self.dataset_npz['E']),-1))
    return np.array(E)

def MD17_extract_F(self):
    if getattr(self,'dataset_npz',None) is None:
        dataset=self.dataset
        F=[]
        l=len(dataset)    
        for i in range(l):
            F.append(dataset[i]['forces'].numpy())
            print(f"Exctracting db forces: {i/l*100:.0f}%",end='\r')

        print("")
    else:
        F=self.dataset_npz['F']
    return np.array(F).reshape(len(F),-1)

def MD17_R_to_dist(self):
    if getattr(self,'dataset_npz',None) is None:
        dataset=self.dataset
        R=[]
        l=len(dataset)    
        for i in range(l):
            R.append(dataset[i]['_positions'].numpy())
            print(f"Exctracting db positions: {i/l*100:.0f}%",end='\r')

        print("")
    else:
        R=self.dataset_npz['R']
    R=toDistance(np.array(R))
    return R

def load_npz_file(self,path,*args):
    self.dataset=np.load(path)

def load_MD17_file_molecule(self,path,molecule=None,npz_data=None):
    if molecule is None:
        print_warning("No molecule selected for MD17 model (no second arg given). Entire dataset taken.")
    from schnetpack.datasets import MD17
    data=MD17(path,molecule=molecule)

    if npz_data is not None:
        self.dataset_npz=np.load(npz_data)
    else:
        self.dataset_npz=None

    self.dataset=data

def get_info_split_train_indices(db,model_path,data_path):
    path='Info/split.npz'
    split=np.load(path)
    return split['train_idx']

def get_info_split_train_indices_preload(db,data_path):
    split=np.load(data_path)
    return split['train_idx']

def get_info_savep_train_indices_preload(db,data_path):
    with open(data_path,'rb') as file:
        save=pickle.loads(file.read())
    return save['training_indices']

def save_split_MD17(db,ind,name):
    dataset=db.dataset
    para=db.para['generate_training_data']
    ind_rest= np.delete(np.arange(len(dataset)),ind)
    val=np.random.choice(ind_rest,para['n_val'],replace=False)
    test=np.delete(ind_rest,val)

    path=os.path.join(db.info_path,'split.npz')
    np.savez(path,train_idx=ind,val_idx=val,test_idx=test)
    db.step_dataset_path=path

def sgdml_dataset_fix_path(db):
    full_path = os.path.dirname(os.path.realpath(__file__))
    data_path=db.args['dataset_path']

    dataset=dict(np.load(data_path))
    dataset['name']=os.path.join(full_path,data_path)
    np.savez_compressed(data_path,**dataset)



