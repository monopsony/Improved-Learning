
# Improved Learning

Requires python 3.6 or higher.

Sample usage:
```bash
python run.py -s 2 -n
```

-s N arguments tells the script to do 2 iteration **steps**
-n arguments tells the script to start a **new** run
The -n argument is currently recommended for *all* uses as resuming has not been extensively tested


## para.py file

The para.py file contains all relevant parameters. Notably, for a default (sGDML) run, the important parameters to change are:

**\[initial_dataset\]**: path to the dataset of interest (from the run directory)
**\[storage_name\]**: name of the folder in which to save results
**\[init_train\]\[args\]**: by default (sgdml) this corresponds to number of training, validation and testing points for the initial model
**\[step_train\]\[args\]**: same as above but for the iterations past the initial model

