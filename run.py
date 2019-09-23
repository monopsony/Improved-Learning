from util import*
import descri,predict,cluster,task_funcs
import numpy as np 
from para import parameters as para
import para as para_file



def load_dataset(dataset_path):
    '''
    Loads the dataset file (.xyz or .npz) and saves it into a dictionary. 
    
    Parameters:
        -dataset_path:
            String corresponding to the path to the dataset file
        
    Returns:
        -data:
            The dataset
                
    '''
    ext=os.path.splitext(dataset_path)[-1]
    
    #xyz file
    if ext==".xyz":
        try:
            file=open(dataset_path)
            dat=read_concat_ext_xyz(file)
            data={ 'R':np.array(dat[0]),'z':dat[1],'E':np.reshape( dat[2] , (len(dat[2]),1) ),'F':np.array(dat[3]) }
        except getopt.GetoptError as err:
            print(err)
            return False
    #npz file        
    elif ext==".npz":
        try:
            data=np.load(dataset_path)
        except Exception as e:
            print_error(str(e))
            
    return data

progress_stages=[
    'init_train',
    'save_init',
    'cluster_init',
    'compute_prediction_error',
    'generate_subset',
    'reclustering',
    'generate_training_data',
    'step_train',
    'save',
]

stage_functions={
    'init_train':task_funcs.init_train_func,
    'cluster_init':task_funcs.cluster_init,
    'save_init':task_funcs.save_file_init,
    'save':task_funcs.save_file,
    'compute_prediction_error':task_funcs.compute_prediction_error,
    'generate_subset':task_funcs.generate_subset,
    'reclustering':task_funcs.recluster,
    'generate_training_data':task_funcs.generate_training_data,
    'step_train':task_funcs.step_train,
}

def load_save_file(arg_dic):
    file_name='Info/save.p'
    path_separator=((os.name=='nt') and "\\") or  "/" 
    path=os.path.dirname(os.path.realpath(__file__))+path_separator+file_name

    if (not os.path.exists(path)) or (arg_dic['new_file']):
        db=data_base(para,arg_dic)

    else:
        with open(path,'rb') as file:
            dic=pickle.loads(file.read())

        db=data_base(para,arg_dic)

        #which stage to start from when reloading
        reload_stage='compute_prediction_error'
        for k in progress_stages:
            if k==reload_stage:
                break
            db.stages[k]=True

        if dic.get('training_indices',False) is not None:
            db.training_indices=dic['training_indices']
        else:
            print_warning('No training indices found in Info/save.p')

        if not (dic['para']==para):
            print_warning("Differences found between save file parameters and current para.py dictionary.")

    return db

def find_valid_path(path):
    n,ori=1,path
    while os.path.exists(path):
        path='{}_{}'.format(ori,n)
        n+=1
        if n>1000:
            print_error("Broke out of 'find_valid_path' after {} iterations (current path: {}). How?".format(n,path))
    return path

class data_base():

    def __init__(self,para,arg_dic):
        self.para=para
        self.args=arg_dic


        self.path_separator=((os.name=='nt') and "\\") or  "/" 
        self.dir_path=os.path.dirname(os.path.realpath(__file__))+self.path_separator
        self.info_path=self.dir_path+"Info"+self.path_separator
        self.storage_path=find_valid_path(self.dir_path+"storage/"+self.para['storage_name'])+self.path_separator


        self.stages={k:False for k in progress_stages}
        self.vars=[]

        self.data_set=load_dataset(para['initial_dataset'])
        data_set=self.data_set

        #get the needed vars ready
        #parses through data set and uses the given functions to generate the needed variables
        #f.e. interatomic distances and energies
        for i in range(len(para['var_funcs'])):
            f_name,f=para['var_funcs'][i],None
            if callable(f_name):
                f=f_name
            else:
                f=(getattr(descri,f_name,None)) or (getattr(para,f_name,None))
                if not callable(f):
                    print("Could not find function or valid function name in para.var_funcs, index {}".format(i))
                    sys.exit(2)

            self.vars.append(f(data_set))

    save_file_name='save.p'
    current_step=0

    def check_current_stage(self):
        stage='compute_prediction_error'
        for i in progress_stages:
            if not self.stages[i]:
                stage=i
                break

        if stage=='compute_prediction_error':
            #proceed to next step, check if enough steps are left to proceed
            self.current_step=self.current_step+1
            print_debug(f"Step {self.current_step} out of {self.args['total_steps']}")
            if self.current_step>self.args['total_steps']:
                return False #gets out of the loop

            for i in range(progress_stages.index('compute_prediction_error')+1,len(progress_stages)):
                self.stages[progress_stages[i]]=False

        self.current_stage=stage
        return True

    def proceed(self):
        if getattr(self,'current_stage',None) is None:
            return

        self.execute_stage(self.current_stage)

    def execute_stage(self,stage):
        print_warning("Executing stage: {}".format(stage))
        if stage_functions.get(stage,None) is None:
            print_error('Tried calling stage_function for stage {}, but was not found. Aborted.'.format(stage))

        stage_functions[stage](self)
        self.stages[stage]=True


    def save_file(self):
        path=self.info_path+self.save_file_name
        model_path,model_new_path=self.info_path+self.para['model_name'],self.info_path+self.para['model_new_name']

        if not os.path.exists(model_new_path):
            print_error("Expected new model at {}, but not found. Please check your train/train_step files. Aborted".format(model_new_path))

        dic={
            'para':self.para,
            'stages':self.stages,
            'training_indices':self.new_training_indices,
        }

        with open(path,'wb') as file:
            pickle.dump(dic,file) 

        N=len(self.new_training_indices)
        it_path="{}indices_{}".format(self.storage_path,N)

        if not os.path.exists(it_path):
            os.makedirs(it_path)

        for i in [model_path,model_new_path,path]:
            if os.path.exists(i):
                shutil.copy(i,it_path)

        #new model becomes model
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(model_new_path):
            shutil.copy(model_new_path,model_path)
            os.remove(model_new_path)

        #new_training_indices become training_indices
        self.training_indices=self.new_training_indices
        self.new_training_indices=None

    def load_model(self):
        #get the prediction funcs ready
        para=self.para['predict_error']
        name=self.para['model_name']
        load_f=find_function(para['load_func'],para_file,predict,globals()) 
        predict_f=find_function(para['predict_func'],para_file,predict,globals()) 

        load_f(self,self.info_path+name)

        self.predict_func=predict_f

    def predict(self,R): 
        pf=self.predict_func
        if (pf is None) or (not callable(pf)):
            print("Predict function not found or not callable.")
            sys.exit(2)
        f=pf

        out=f(self,R)
        return out

    def calculate_errors(self):
        
        print("Calculating errors")
        
        para=self.para['predict_error']

        #helping variables
        cluster_indices=self.init_cluster_indices
        n_clusters=len(cluster_indices)

        for i in range(1): #IN CASE we want to calculate more errors further down the line (taken from cluster-error)

            error_func=find_function(para['error_func'],para,globals(),descri)
            
            input_values=self.vars[para['input_var_index']]
            comparison_values=self.vars[para['comparison_var_index']]
            predict_values=self.predict(input_values)

            err=error_func(predict_values,comparison_values)
            mse=err.mean()

            self.sample_err=err
            self.cluster_err=[ err[x].mean() for x in cluster_indices]

def parse_arguments(argv):
    '''
    This function parses through the arguments and acts accordingly.
    '''

    dic={
        'total_steps':0,
        'new_file':False,
    }

    
    try:
        opts,args=getopt.getopt(argv,"hs:n")
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    
    
    for opt,arg in opts:
        if opt=='-h':
            print_help()
            sys.exit()
            
        elif opt=="-s":
            dic['total_steps']=int(arg)

        elif opt=='-n':
            dic['new_file']=True
    
    return dic

if __name__=='__main__':

    arg_dic=parse_arguments(sys.argv[1:])

    if arg_dic['new_file']:
        info_path="Info/"
        #remove all files in Info
        #if arg -n was picked
        for file in os.listdir(info_path):
            file_path=os.path.join(info_path,file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    #db=data_base(para)
    db=load_save_file(arg_dic)

    while_count,while_break=0,False
    while (db.check_current_stage()) and (not while_break):
        while_count+=1
        if while_count>999:
            while_break=True
        db.proceed()

    if while_break:
        print_error("Broke out of while loop in run.py main. Something went wrong")
    else:
        print_blue("Exited without error.")
    







