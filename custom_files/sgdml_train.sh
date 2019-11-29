#!/bin/bash -l

printf "\n In sgdml_train, %s %s %s %s" "$1,$2,$3,$4,$5,$6"

sgdml all -o -v $2 -t $2 $1 $4 $5 $6 >> out.txt

mv ${3}*?train*?${4}*?sym*?.npz Info/model_new 