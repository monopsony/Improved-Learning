import numpy as np
import predict,cluster,descri
import para as para_file
from util import*

def move_init_model(self):
    path=self.info_path+self.para['model_new_name']
    init_path=self.args['init_model']

    if not os.path.exists(init_path):
        print_error(f'In task_func move_init_model: {init_path} is not a valid path. Aborted.')
    else:
        shutil.copyfile(init_path,path)

def init_train_func(self):
    para=self.para['init_train']

    if self.args['init_model']:
        move_init_model(self)
    else:
        run_custom_file(self,para['file'],para['args'])

    if self.args['init_indices']:
        initial_indices_func=find_function(para['initial_indices_preload'],para_file,predict,descri)
        self.new_training_indices=initial_indices_func(self,self.args['init_indices'])
    else:
        initial_indices_func=find_function(para['initial_indices'],para_file,predict,descri)
        self.new_training_indices=initial_indices_func(self,os.path.join(self.info_path,self.para['model_new_name']),self.main_dataset_path)

def save_file_init(self):
    self.save_file()

def save_file(self):
    self.save_file()

def cluster_init(self):

    cluster_path=self.args['cluster_path']
    if cluster_path is None:
        self.init_cluster_indices=cluster.cluster_do(self,None,"init_cluster")

    else:
        if not os.path.exists(cluster_path):
            print_error(f"No file found under {cluster_path}.")

        cl_ind=np.load(cluster_path,allow_pickle=True)
        var_index=self.para['clusters'][0]['var_index']
        N=len(self.vars[var_index])
        if N!=len(np.concatenate(cl_ind)):
            print_warning(
                f"Given cluster file might contain different amount of points than given dataset. Length of vars as given by var_index {var_index}: {N}. Number of indices in given cluster file: {len(np.concatenate(cl_ind))}.")

        self.init_cluster_indices=cl_ind

def compute_prediction_error(self):
    self.load_model()
    self.calculate_errors()

def load_dataset(self):
    para=self.para['load_dataset']

    if para.get('prepare_dataset',None) is not None:
        func=find_function(para['prepare_dataset'],para_file,descri,globals())
        func(self)

    if len(para['args'])>0:
        self.main_dataset_path=para['args'][0]
    else:
        self.main_dataset_path=None

    f=find_function(para['func'],para_file,globals(),descri,predict)
    f(self,*generate_custom_args(self,para['args']))

    #get the needed vars ready
    #parses through data set and uses the given functions to generate the needed variables
    #f.e. interatomic distances and energies
    for i in range(len(para['var_funcs'])):
        f=find_function(para['var_funcs'][i],para_file,descri,globals())
        self.vars.append(f(self))  

def recluster(self):
    self.recluster_indices=cluster.cluster_do(self,self.subset_ind,'reclustering') #second parameter determines the initial subset indices

def generate_subset(self):  
    para=self.para['generate_subset']
    func_name=para['func']
    func=find_function(func_name,para_file,cluster,globals(),descri)
    self.subset_ind=func(self,*para['args'])

def generate_training_data(self):
    cl_ind,err=self.recluster_indices,self.sample_err
    para=self.para['generate_training_data']

    #generate new indices
    indices_func=find_function(para['indices_func'],para_file,globals(),descri,cluster)
    ind=indices_func(self,cl_ind,err)

    self.new_training_indices=np.concatenate([self.training_indices,ind])

    if para['save_func'] is not None:
        #create new dataset to train on
        save_func=find_function(para['save_func'],para_file,globals(),descri,predict,cluster)
        save_func(self,self.new_training_indices,self.para['step_dataset_name'])

def step_train(self):
    para=self.para['step_train']

    if self.current_step==self.args['total_steps']:
        args=para.get('last_step_args',para['args'])
    else:
        args=para['args']

    args=[getattr(self,'step_dataset_path',None)]+args
    run_custom_file(self,para['file'],args)

    path=self.info_path+self.para['model_new_name']
    if not os.path.exists(path):
        print_error("No model found at path {} after step_train.\nPlease check your step_train file ({})".format(path,para['file']))


