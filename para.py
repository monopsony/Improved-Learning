parameters={

	'model_name':'model', #not actually a parameter atm, dont touch
	'model_new_name':'model_new', #idem dont touch
	'step_dataset_name':'dataset', #idem dont touch

	'storage_name':'uracil_test', #name of the folder in which to save results

	'load_dataset':{
		'func':'load_MD17_file_molecule',  #load_MD17_file_molecule
			#args are [db object, args below]
		'args':['func:get_d_arg',None,'datasets/ethanol_db_E_F_R.npz'], #get_d_arg gets the -d argument given 

		'var_funcs':{
			0:'MD17_R_to_dist',
			1:'MD17_extract_E',
			2:'MD17_extract_F',
			3:'db_indices',
		}, #end of 'var_funcs'
	},

	'init_train':{
		'prepare_dataset':None,
								#function called before the training session to 'prepare' dataset.
								#can be set to None if no preparation is needed (prining dataset info/making a backup/whatever)
								#.py or .sh
								#.py NEEDS a run() function to which one passes the needed arguments

		'prepare_dataset_args':[],
								#if (additional) args are needed, define them here
								#first arg is always the database object

		'file':'MD17_train_init.py',
								#needs to create a model file of the same name 
								#as the one given by the model_name parameter
								#args are given below

		'args':['func:get_d_arg',200,1000,500],
						   		#passed to the .sh script / .py run function
						   		#(commonly needs to pass the name of the dataset, f.e. as given by -d with 'func:get_d_arg')
						   		#function outputs can be passed by passing 'func:{func_name}'
						   		#the actual argument passed will be the (first) return of that function call
						   		#Argument passed to the these custom functions (inside 'args') is just the db object

		'initial_indices':'get_info_split_train_indices',
								#the function here needs to return a list that includes all indices 
								#(corresponding to the initial data set) that were used to create the initial model
								#the arguments given are the db, the path to the model and path to the initial data set


		'initial_indices_preload':'get_info_split_train_indices_preload',

	}, #end of 'init_train'

	'clusters':{
		'init_cluster':[0,1],
		'reclustering':[2],

		#indices below define clustering schemes, but don't do anything on their own
		#init_cluster and reclustering parameters above are list of indices
		#indices correspond to clustering schemes below, applied successively
		#e.g. if 'init_cluster':[0,1], the initial clusterisation algorithm will consist of separating the initial dataset
		#according to scheme 0 and then the resulting clusters will be re-clustered according to scheme 1
		0:{
		    'type':'Agglomerative', #types: Agglomerative, Kmeans
		    'n_clusters':12,
		    'initial_number':10000, 
		    'distance_matrix_function':'distance_matrix_euclidean',
		    'linkage':'complete',
		    'cluster_choice_criterion':'smallest_max_distance_euclidean',
		    'var_index':0,
		    },

		1:{
			'type':'Kmeans',
			'n_clusters':5,
			'var_index':1,
			},

		2:{
		    'type':'Agglomerative',
		    'n_clusters':400,  ##func name (string) or flat number
		    #'n_clusters_args':[400],
		    'initial_number':10000, #1 means the entire subset
		    'distance_matrix_function':'distance_matrix_euclidean',
		    'linkage':'complete',
		    'cluster_choice_criterion':'smallest_max_distance_euclidean',
		    'var_index':0,
		    },
	}, #end of 'clusters'

	'predict_error':{
		'load_func':'load_model_MD17',
		'predict_func':'predict_F_MD17',
		'input_var_index':3,
		'comparison_var_index':2,
		'error_func':'mean_squared_error_sample_wise',
		'file_name':'mean_squared_error',
	}, #end of 'predict_error'

	'generate_subset':{
		'func':'cluster_above_mse', #(self,*args)
		'args':[1],
	}, #end of 'generate_subset'

	'generate_training_data':{
		'indices_func':'within_cluster_weighted_err_N', #args given are db, recluster indices and sample-wise error array
		'n_points':200,
		'save_func':'save_split_MD17', #args given are db and indices as given by the indices_func (can be None) and step_dataset_name
		'n_val':1000, #only used by f.e. save_split_MD17 when needed to pre-create validation set

	},

	'step_train':{
		'file':'MD17_train.py', #first argument passed is (full) path to the new dataset, created by step above (save_func)
								 #second argument is (full) path to the initial dataset
								 #further arguments are defined below
								 #must save the model to Info/model_name where model_name is given by 'model_new_name' parameter (top)

		'args':['func:get_d_arg',None,None,500], 
								#Argument passed to the custom functions (inside 'args') is just the db object
	},#end of 'step_train'

} #end of parameters


def len_new_training_indices(db):
	return len(db.new_training_indices)


def get_d_arg(self):
	d=self.args['dataset_path']
	return d






