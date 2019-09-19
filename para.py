parameters={

	'model_name':'model.npz', #not actually a parameter atm, dont touch
	'model_new_name':'model_new.npz', #idem dont touch
	'step_dataset_name':'dataset.npz', #idem dont touch

	'initial_dataset':'datasets/uracil.npz',
	'storage_name':'uracil_test', #name of the folder in which to save results

	'init_train':{
		'prepare_dataset':'sgdml_dataset_fix_path.py',
								#function called before the training session to 'prepare' dataset.
								#can be set to None if no preparation is needed (prining dataset info/making a backup/whatever)
								#.py or .sh
								#.py NEEDS a run() function to which one passes the needed arguments

		'prepare_dataset_args':[],
								#if (additional) args are needed, define them here
								#first arg actually passed is always the (full) path to the data set

		'file':'sgdml_train_init.sh',
								#needs to create a model file of the same name 
								#as the one given by the model_name parameter
								#args are given below

		'args':[200,1000,10000],
								#note: first argument passed to the function is ALWAYS the path to the data set
								#in other words, parameters['init_train']['args'][0] is the SECOND argument
						   		#to the .sh script / .py run function
						   		#function outputs can be passed by passing 'func:{func_name}'
						   		#the actual argument passed will be the (first) return of that function call
						   		#Argument passed to the these custom functions (inside 'args') is just the db object

		'initial_indices':'sgdml_initial_indices',
								#the function here needs to return a list that includes all indices 
								#(corresponding to the initial data set) that were used to create the initial model
								#the arguments given are the db, the path to the model and path to the initial data set
	}, #end of 'init_train'

	'var_funcs':{
		0:'r_to_desc',
		1:'extract_E',
		2:'extract_R_concat',
		3:'extract_F_concat',
	}, #end of 'var_funcs'

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
		    'n_clusters':10,
		    'initial_number':5000,
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
		    'initial_number':1, #1 means the entire subset
		    'distance_matrix_function':'distance_matrix_euclidean',
		    'linkage':'complete',
		    'cluster_choice_criterion':'smallest_max_distance_euclidean',
		    'var_index':0,
		    },
	}, #end of 'clusters'

	'predict_error':{
		'load_func':'load_model_sgdml',
		'predict_func':'predict_F_sgdml',
		'input_var_index':2,
		'predict_func_index':0,
		'comparison_var_index':3,
		'error_func':'mean_squared_error_sample_wise',
		'file_name':'mean_squared_error',
	}, #end of 'predict_error'

	'generate_subset':{
		'func':'worst_N_clusters', #(self,err,mse,cl_ind)
		'args':[10],
	}, #end of 'generate_subset'

	'generate_training_data':{
		'indices_func':'within_cluster_lowest_variance', #args given are db, recluster indices and sample-wise error array
		'var_index':0, #lowest variance with respect to what?
		'save_func':'save_sgdml_npz_data', #args given are db and indices as given by the indices_func
	},

	'step_train':{
		'file':'sgdml_train.sh', #first argument passed is (full) path to the new dataset, created by step above (save_func)
								 #second argument if (full) path to the initial dataset
								 #further arguments are defined below
								 #must save the model to Info/model_name where model_name is given by 'model_new_name' parameter (top)

		'args':['func:npz_dataset_get_size',1000,10000], 
								#Argument passed to the custom functions (inside 'args') is just the db object
	},#end of 'step_train'

} #end of parameters


def npz_dataset_get_size(db):
	return len(db.new_training_indices)











