#!/bin/bash -l

printf "\n In sgdml_train, %s %s %s %s" "$1,$2,$3,$4,$5"

sgdml all -o -v $2 -t $2 $1 $3 $4 $5 >> out.txt

mv *?-train*?-sym*?.npz Info/model_new.npz