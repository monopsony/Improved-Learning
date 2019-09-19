import numpy as np
import math as m
import random as rand
import os as os
import hashlib as hashlib

path = os.path.dirname(os.path.realpath(__file__))

def saveData(data, indices, name):
  
  if isinstance(data,basestring):
    database = np.load(data)
  else:
    database=data
  
  energy, coord, forces, elem = database['E'], database['R'], database['F'], database['z']

  if indices:
    base_vars = {'type': 'd',
                 'R': coord[indices],
                 'z': elem,
                 'E': energy[indices],
                 'F': forces[indices],
                 'name': name,
                 'theory': 'unknown'}
  else:
    base_vars = {'type': 'd',
                 'R': coord,
                 'z': elem,
                 'E': energy,
                 'F': forces,
                 'name': name,
                 'theory': 'unknown'}
                 
  md5_hash = hashlib.md5()
  for key in ['z', 'R', 'E', 'F']:
    md5_hash.update(hashlib.md5(base_vars[key].ravel()).digest())
          
  base_vars['md5'] = md5_hash.hexdigest() 
  
  np.savez_compressed(name, **base_vars)

dataset=path+"/Datasets/azobenzene_dft.npz"
dataset_name="azobenzene_i100_s300"
ds=np.load(dataset)
n=ds["R"].shape[0]
#k=n/nChop

#a=range(n)
#indices=rand.sample(a,k)
#indices2=np.delete(a,indices)
name=path+"/SetNew4/dataset.npz"
name2=path+"/SetNew4/Info/dataset_name"

#save dataset
if os.path.exists(name):
  os.remove(name)  
saveData(dataset,None,name)

#save name
f=open(name2,"w")
f.write(dataset_name)




