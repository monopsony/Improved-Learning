import numpy as np
import cluster,descri
from util import*

def load_model_sgdml(self,model_path):

    '''
    This function is called once during initialisation. Any one-time loading
    that needs to be done should be done here. If nothing needs loading, this
    function still needs to exist but can just return immediately.
    
    Parameters:
      -model_path: 
          string of the path to the model file
    '''
    from sgdml.predict import GDMLPredict

    try:
        model=np.load(model_path)
    except:
        print_error("Could not load model file. Please ensure it is of the right format (.npz). Aborted.")

    try:
        gdml=GDMLPredict(model)
        self.gdml=gdml
    except:
        print_error("Unable to read GDML model file. Aborted.")

def sgdml_initial_indices(self,model_path,data_path):

    try:
        model=np.load(model_path)
    except:
        print_error("Could not load model file. Please ensure it is of the right format (.npz). Aborted.")
        sys.exit(2)

    return model['idxs_train']

def predict_F_sgdml(self,cluster_R):
    '''
    Function used to predict forces of data chunks. Pray pay attention to the dimensions.
    The values predicted here are equivalent to those present in the ['F'] key of the data.
    The input is an array containing the spatial information of every sample in a single 
    cluster.
    The output is an array containing the output values (f.e. forces) for every sample in 
    the cluster (same order as the input). 

    The output values are to be given in a 1D single array, so that the function return:
    cluster_F= [[F11,F12,...,F1M],[F21,F22,...,F2M],...,[FN1,FN2,...,FNM]] 
    for a cluster containing N samples and a model output of a total of M values. 
    
    
    Paramters:
        - cluster_R: numpy array containing sample positions. 
                     Dimensions are (n_samples_in_cluster,n_atoms,n_dimensions)
        
    Returns:
        - cluster_F: numpy array containing predicted forces.   
                     Dimensions are (n_samples_in_cluster,n_atoms*n_dimensions)
    '''
    gdml=self.gdml
    
    _,cluster_F=gdml.predict(cluster_R)

    return cluster_F

def predict_E_sgdml(self,cluster_R):
    '''
    See predict. Note:
    - The values predicted here are equivalent to those present in the ['E'] key of the data.
    - This function is used instead of predict (above) if the -n argument is used when calling errors.py
    '''

    gdml=self.gdml
    
    n_samples,n_atoms,n_dim=cluster_R.shape
    cluster_R=np.reshape(cluster_R,(n_samples,n_atoms*n_dim))  
    
    
    cluster_E,_=gdml.predict(cluster_R)

    return cluster_E

def save_sgdml_npz_data(self,ind,name):
    dataset=self.dataset
    tbl={}
    for k,v in dataset.items():
        if k=='F' or k=='R' or k=='E':
            tbl[k]=v[ind]
        else:
            tbl[k]=v 

    np.savez_compressed(os.path.join(self.info_path,name),**tbl)
    self.step_dataset_path=os.path.join(self.info_path,name)

def load_model_MD17(self,model_path):
    import torch
    try:
        model=torch.load(model_path,map_location='cpu') #cba with cuda 
    except Exception as e:
        print(e)
        sys.exit(2)

    self.MD17_model=model

def predict_E_MD17(self,indices):
    m=self.MD17_model
    test=self.dataset.create_subset(indices)

    import schnetpack as spk 
    test_loader=spk.AtomsLoader(test,batch_size=1000)
    preds=[]
    for count,batch in enumerate(test_loader):
        #TBA currently only supports single batch
        preds.append(m(batch)['energy'].detach().cpu().numpy())

    return np.concatenate(preds)

def predict_F_MD17(self,indices):
    m=self.MD17_model
    test=self.dataset.create_subset(indices)
    import schnetpack as spk 
    test_loader=spk.AtomsLoader(test,batch_size=1000)
    preds=[]
    for count,batch in enumerate(test_loader):
        #TBA currently only supports single batch
        preds.append(m(batch)['forces'].detach().cpu().numpy())

    F=np.concatenate(preds)
    F=F.reshape(len(F),-1)
    return F
