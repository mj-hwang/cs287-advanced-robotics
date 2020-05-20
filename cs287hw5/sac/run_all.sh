#! /bin/bash

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reinf -e 3

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reparam -e 3 --reparameterize

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reparam_2qf -e 3 --reparameterize --two_qf


python train_mujoco.py --env_name Ant-v2 --exp_name reinf -e 3

python train_mujoco.py --env_name Ant-v2 --exp_name reparam -e 3 --reparameterize

python train_mujoco.py --env_name Ant-v2 --exp_name reparam_2qf -e 3 --reparameterize --two_qf


python train_mujoco.py --env_name Hopper-v2 --exp_name reinf -e 3

python train_mujoco.py --env_name Hopper-v2 --exp_name reparam -e 3 --reparameterize

python train_mujoco.py --env_name Hopper-v2 --exp_name reparam_2qf -e 3 --reparameterize --two_qf