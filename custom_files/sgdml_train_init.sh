#!/bin/bash -l

printf "\n In sgdml_train_init, %s %s %s %s" "$1,$2,$3,$4,$5"

sgdml all -o $1 $3 $4 $5 >> out.txt

mv ${2}*?train*?${3}*?sym*?.npz Info/model_new  