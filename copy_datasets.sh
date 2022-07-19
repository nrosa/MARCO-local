#!/bin/bash
local_rank=$(($SLURM_PROCID % $SLURM_NTASKS_PER_NODE))

data_set=$1

if [[ $local_rank -eq 0 ]]
then
    cd $LOCALDIR
    cp "/scratch1/ros282/$data_set.tar" "$data_set.tar"
    tar xf "$data_set.tar"

    cp "/scratch1/ros282/c3_testset.tar" c3_testset.tar
    tar xf c3_testset.tar
fi

cd /home/ros282/projects/marco-retrain/

