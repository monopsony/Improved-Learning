#!/bin/bash -l

printf "\n In sgdml_train_init, %s %s %s %s" "$1,$2,$3,$4"

sgdml all -o $1 $2 $3 $4 >> out.txt

mv *?-train*?-sym*?.npz Info/model_new.npz