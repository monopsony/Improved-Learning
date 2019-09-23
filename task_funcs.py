import numpy as np
import predict,cluster,descri
import para as para_file
from util import*


def init_train_func(self):
    para=self.para['init_train']
    if para['prepare_dataset'] is not None:
        run_custom_file(self,para['prepare_dataset'],[self.dir_path+self.para['initial_dataset']]+para['prepare_dataset_args'])
    run_custom_file(self,para['file'],[self.para['initial_dataset']]+para['args'])
    initial_indices_func=find_function(para['initial_indices'],para_file,predict,descri)
    self.new_training_indices=initial_indices_func(self,self.info_path+self.para['model_new_name'],self.para['initial_dataset'])

def save_file_init(self):
    self.save_file()

def save_file(self):
    self.save_file()

def cluster_init(self):
    self.init_cluster_indices=cluster.cluster_do(self,None,'init_cluster') #second parameter determines the initial subset indices

def compute_prediction_error(self):
    self.load_model()
    self.calculate_errors()

def recluster(self):
    self.recluster_indices=cluster.cluster_do(self,self.subset_ind,'reclustering') #second parameter determines the initial subset indices

def worst_N_clusters(self,N,*args):
    mse=self.cluster_err
    cl_ind=self.init_cluster_indices
    sorted_ind=np.argsort(mse)
    clusters=np.array(cl_ind)[sorted_ind[-N:]]
    ind=np.concatenate(clusters)
    return ind

def generate_subset(self):  
    para=self.para['generate_subset']
    func_name=para['func']
    func=find_function(func_name,para_file,globals(),descri)
    self.subset_ind=func(self,*para['args'])

def generate_training_data(self):
    cl_ind,err=self.recluster_indices,self.sample_err
    para=self.para['generate_training_data']

    #generate new indices
    indices_func=find_function(para['indices_func'],para_file,globals(),descri,cluster)
    ind=indices_func(self,cl_ind,err)

    self.new_training_indices=np.concatenate([self.training_indices,ind])

    #create new dataset to train on
    save_func=find_function(para['save_func'],para_file,globals(),descri,predict,cluster)
    save_func(self,self.new_training_indices)

def step_train(self):
    para=self.para['step_train']
    args=[
        self.info_path+self.para['step_dataset_name'],
        self.dir_path+self.para['initial_dataset'],
    ]+para['args']

    run_custom_file(self,para['file'],args)

    path=self.info_path+self.para['model_new_name']
    if not os.path.exists(path):
        print_error("No model found at path {} after step_train.\nPlease check your step_train file ({})".format(path,para['file']))


