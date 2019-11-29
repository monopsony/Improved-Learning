from util import*
import predict,descri
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances


def smallest_max_distance_euclidean(sample,clusters):
    '''
    Finds the cluster that the given sample belongs to. Simple euclidean distance is used.
    (The metric used should be the same as for the agglomerative clustering.)
    
    Paramters:
        -sample: 
            numpy array containing positions of atoms of one samples
            Dimensions: (n_atoms,n_dimensions)
        
        -clusters: 
            numpy array containing positions within each cluster
            Dimensions: (n_clusters,n_atoms,n_dimensions)
                    
    Returns:
        -index of cluster that the sample belongs to / closest cluster
                                         
    '''
    
    g=np.zeros(len(clusters))
    for i in range(len(clusters)):
        g[i]=np.max(np.sum(np.square(clusters[i]-sample),1))   #numpy difference=>clusters[c]-sample elementwise for each c
    return np.argmin(g)
    
def distance_matrix_euclidean(data):

    return euclidean_distances(data,data)

cluster_funcs={}

def agglomerative_clustering(data_set,indices,clustering_index):
    para=data_set.para['clusters'][clustering_index]
    f_name=para["distance_matrix_function"]
    matrix_f=find_function(f_name,para,globals(),descri)

    data,N,ind_cluster,cluster_data=None,None,[],[]

    ##get n_clusters
    n_clusters_para=para['n_clusters']
    if isinstance(n_clusters_para,basestring):
        nc_func=find_function(para['initial_number'],para,globals(),descri)

    else:
        n_clusters=n_clusters_para


    ##generate data
    if not (indices is None):
        data=data_set.vars[para['var_index']][indices]
    else:
        data=data_set.vars[para['var_index']]




    ##get number of initial clusters
    if isinstance(para['initial_number'],basestring):
        init_n_func=find_function(para['initial_number'],para,globals(),descri)
        if callable(init_n_func):
            N=init_n_func(self)
        else:
            print_error("In agglomerative_clustering: initial_number function {} not callable. Aborted.".format(init_n_name))

    elif para["initial_number"]>1:
        N=para["initial_number"]
    else:
        N=len(data)*para["initial_number"]


    ##prepare agglomerative vars
    ind_all=np.arange(len(data))
    ind_init=np.random.permutation(ind_all)[:N] 
    data_init=data[ind_init]
    ind_rest=np.delete(ind_all,ind_init)
    data_rest=data[ind_rest]
    M=matrix_f(data_init)


    ##agglo
    print_blue("Start Agglomerative Clustering")
    cinit_labels=AgglomerativeClustering(affinity="precomputed",n_clusters=n_clusters,linkage=para.get('linkage','complete')).fit_predict(M)
    print_blue("Agglomerative Clustering done")

    cluster_ind=[]
    for i in range(n_clusters):
        ind=np.concatenate(np.argwhere(cinit_labels==i))

        #convert back to initial set of indices
        ind=ind_init[ind]

        cluster_ind.append(ind.tolist())
        cluster_data.append(np.array(data[cluster_ind[i]]))



    #divide rest into clusters
    #using para['cluster_choice_criterion']
    #+ni to find the old index back from entire dataset
    #print("Clustering rest of data...")
    #outs=np.trunc(np.linspace(0,len(data_rest),99))
    choice_func=find_function(para["cluster_choice_criterion"],para,globals(),descri)



    for i in range(len(data_rest)):
        #%done output
        # if i==outs[0]:
        #     if i==0:
        #         ch=0
        #     else:
        #         ch=float(i)/len(data_rest)*100
        #     sys.stdout.write("\r[{:.0f}%] done".format(ch+1))
        #     sys.stdout.flush()
        #     outs=np.delete(outs,0)
    
        c=choice_func(data_rest[i],cluster_data) #c is the cluster# it belongs to
        cluster_ind[c].append(ind_rest[i])


    if indices is None:
        return cluster_ind

    #if needed, change the indices of every cluster back corresponding to original data set
    for cl in cluster_ind:
        for i in range(len(cl)):
            cl[i]=indices[cl[i]]

    return cluster_ind
cluster_funcs["Agglomerative"]=agglomerative_clustering

def kmeans_clustering(data_base,indices,clustering_index):
    para=data_base.para['clusters'][clustering_index]
    data,n_clusters,cluster_ind=None,para["n_clusters"],[]
    if not (indices is None):
        data=data_base.vars[para["var_index"]][indices]
    else:
        data=data_base.vars[para["var_index"]]


    cluster_labels=KMeans(n_clusters=n_clusters,init="k-means++").fit_predict(data)

    for i in range(n_clusters):
        ind=np.concatenate(np.argwhere(cluster_labels==i).tolist())
        #convert back to initial set of indices
        #no need here
        cluster_ind.append(ind)

    if indices is None:
        return cluster_ind

    #if needed, change the indices of every cluster back corresponding to original data set
    for cl in cluster_ind:
        for i in range(len(cl)):
            cl[i]=indices[cl[i]]

    return cluster_ind
cluster_funcs["Kmeans"]=kmeans_clustering

def cluster_do(self,init_indices,para_ind):
    para=self.para['clusters']
    cluster_para_ind=para.get(para_ind,None)
    n_clusters=((not cluster_para_ind is None) and len(cluster_para_ind)) or 0
    if n_clusters==0:
        return 

    #perform first clusterisation
    cl_type=para[cluster_para_ind[0]]['type']
    cl_func=cluster_funcs.get(cl_type,None)
    if not cl_func:
        print_error("Cluster functions of type '{}'' not found. Check your parameters.".format(cl_type))
    cl_ind=cl_func(self,init_indices,cluster_para_ind[0])



    #perform further clusterisations  
    for i in cluster_para_ind[1:]:
        cl_ind_new=[]
        for cl in cl_ind:
            cl_type=para[i]['type']
            cl_func=cluster_funcs.get(cl_type,None)
            if not cl_func:
                print_error("Cluster functions of type '{}' not found. Check your parameters.".format(cl_type))

            cl_cl_ind=cl_func(self,cl,i)
            for j in cl_cl_ind:
                cl_ind_new.append(j)

        cl_ind=cl_ind_new

    return cl_ind

def worst_N_clusters(self,N,*args):
    mse=self.cluster_err
    cl_ind=self.init_cluster_indices
    sorted_ind=np.argsort(mse)
    clusters=np.array(cl_ind)[sorted_ind[-N:]]
    ind=np.concatenate(clusters)
    return ind

def cluster_above_mse(self,fact,*args):
    mse=np.array(self.cluster_err)
    mmse=np.mean(mse)
    cl_ind=self.init_cluster_indices
    cl_ind_new=np.concatenate(np.argwhere(mse>mmse*fact))
    print_debug(f"adding {len(cl_ind_new)} clusters in cluster_above_mse")
    clusters=np.array(cl_ind)[cl_ind_new]
    ind=np.concatenate(clusters)
    return ind

