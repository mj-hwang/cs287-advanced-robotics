# Project setup script
# Source this file to set up the environment for this project.


ENV_NAME='cs294drl_hw5_sac'

if [ "$1" = "setup" ]; then
    echo "Creating conda environment..."
    conda env create -f environment.yml
elif [ "$1" = "remove" ]; then
    conda remove --name $ENV_NAME --all --yes
else

    export PROJECT_HOME="$(pwd)"
    
    alias ph="cd $PROJECT_HOME"
    
    
    alias set_display="export DISPLAY=':0.0'"
    alias unset_display="unset DISPLAY"
    
    export MPLBACKEND='Agg'
    
    source activate $ENV_NAME
    
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/young/.mujoco/mjpro150/bin"
fi
