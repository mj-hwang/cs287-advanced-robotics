#! /bin/bash

function filter_experiment_dirs {
   ls data | grep -e $1 | sed -e 's/^/data\//'
}

function filter_experiment_config {
   ls data | grep -e $1 | sed -e "s/$2.*//"
}


mkdir -p plots

python myplot.py  \
    --legend $(filter_experiment_config 'sac_HalfCheetah' '\d{2}-\d{2}-\d{4}') \
    --title 'HalfCheetah SAC' \
    --output plots/HalfCheetah_SAC.png \
    $(filter_experiment_dirs 'sac_HalfCheetah')
    
python myplot.py  \
    --legend $(filter_experiment_config 'sac_Ant' '\d{2}-\d{2}-\d{4}') \
    --title 'Ant SAC' \
    --output plots/Ant_SAC.png \
    $(filter_experiment_dirs 'sac_Ant')
    