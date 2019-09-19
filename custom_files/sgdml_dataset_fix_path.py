import numpy as np
import os

def run(data_path):
  path = os.path.dirname(os.path.realpath(__file__))
  path_separator=((os.name=='nt') and "\\") or  "/" 
  parent = os.path.dirname(path)

  dataset=dict(np.load(data_path))
  dataset['name']=parent+path_separator+os.path.basename(data_path)
  np.savez_compressed(data_path,**dataset)

