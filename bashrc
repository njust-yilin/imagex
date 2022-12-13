#! /bin/bash

source ~/miniconda3/bin/activate imagex

export IMAGEX_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export PYTHONPATH=$PYTHONPATH:$IMAGEX_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/envs/imagex/lib
export PYTHONUNBUFFERED=1

# There appear to be 4 leaked shared_memory objects to clean up at shutdown
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
