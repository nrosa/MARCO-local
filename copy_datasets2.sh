#!/bin/bash
local_rank=$(($SLURM_PROCID % $SLURM_NTASKS_PER_NODE))

data_set1=$1
data_set2=$2

if [[ $local_rank -eq 0 ]]
then
    cd $LOCALDIR
    cp "/scratch1/ros282/$data_set1.tar" "$data_set1.tar"
    tar xf "$data_set1.tar"

    cp "/scratch1/ros282/$data_set2.tar" "$data_set2.tar"
    tar xf "$data_set2.tar"
fi

cd /home/ros282/projects/marco-retrain/

