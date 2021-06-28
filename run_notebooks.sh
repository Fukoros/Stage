#!/bin/bash

# This script updates the covid-19-community knowledge graph
 
LOGDIR="res"/`date +%Y-%m-%d-%H-%M`
mkdir -p "$LOGDIR"
echo "$LOGDIR"
# enable conda in bash (see: https://github.com/conda/conda/issues/7980)
eval "$(conda shell.bash hook)"

# run Jupyter Notebooks to download, clean, and standardize data for the knowledge graph
# To check for any errors, look at the executed notebooks in the $LOGDIR directory

for f in T*.ipynb 
do 
  echo "Processing $f file.."
#   papermill $f "$LOGDIR"/$f
done

# deactivate conda environment
conda deactivate 