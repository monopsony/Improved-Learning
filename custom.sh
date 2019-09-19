#!/bin/bash -l
#SBATCH -n 28
#SBATCH --ntasks-per-node=28
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=00:30:00
#SBATCH -p batch
#SBATCH --qos=qos-batch
#SBATCH -J tr_it


python3 run.py -s 2 -n
wait 
