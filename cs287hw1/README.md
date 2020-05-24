# CS287 Homework 1
## Setup
The code is written in python 3. In order to install the requiriments run 

`pip install -r requiriments`

Before running the code make sure to set the `PYTHONPATH` to be the homework folder. You can do that by running 
the following command:

`export PYTHONPATH=path_to_cs287_hw1/`

The data and logging for all the parts experiments will be saved in the folders `data/partX/`. This will contain
the final value function contour (if the state space is of dimension lower or equal than 2), the learning curve of the
run, a csv file with the returns per iteration, and a json file with the parameters of the run. If the experiments are
 run rendering the progress (by using the flag `-r` or `--render`), then it will save two videos with the learning
 progess, one for the contours and another with the rollouts.
 
 
To visualize data of combined experiments you can use the command `python viskit/frontend 
path_to_the_experiments_folder`. You will be able to visualize the learning curves split by any parameter that you 
decide or we ask in the report.

Each part folder has its own run script named `partX/run_partX.py` that will run all the environments we ask for.
You will have to modify the arguments of the run script to answer the different questions, you can see what each
command means by running `python partX/run_partX.py --help`. In all the run scripts you can run visualize the 
rollouts by adding the flag `--render` or `-r` ,i.e., `python partX/run_partX.py -r`.

The functions are documented to explain all the inputs, returns, and the utilities needed to solve each question.


## Part 1: Value Iteration
### Question (a): Implement value iteration for a deterministic policy
You will need to fill the code in `part1/tabular_value_iteration.py`
 below the lines `if self.policy_type == 'deterministic'`. Run `python part1/run_part1.py `
to run the code that will generate the plots and loggings.

### Question (b): Implement value iteration for a maximum entropy policy
You will need to fill the code in `part1/tabular_value_iteration.py`
 below the lines `if self.policy_type == 'max_ent'`. Run `python part1/run_part1.py -p max_ent -t TEMPERATURE `
to run value iteration with a maximum entropy policy and different temperature values.


## Part 2: Discretization
### Question (a): Implementation of nearest-neighbor interpolation
You will need to fill the code in `part2/discretize.py` below the lines `if self.mode == 'nn'`. To run this question 
type `python part2/run_part2_ab.py -m nn `. You can modify the number of points used per dimension in the state
anc action space by running
`python part2/run_part2_ab.py -m nn -s NUM_STATES`, where `NUM_STATES` is the number of 
points per state dimension.


### Question (b): Implementation of n-linear interpolation
You will need to fill the code in `part2/discretize.py` below the lines `if self.mode == 'linear'`. To run this question 
type `python part2/run_part2_ab.py -m linear -s NUM_STATES` in the command line.
  
  
### Question (c): Nearest-neighbor vs. n-linear interpolation
To run this question 
type `python part2/run_part2_c.py -m INTERPOLATION_MODE`, where `INTERPOLATION_MODE` is either `nn` for 
nearest-neighbor interpolation, or `linear` for n-linear interpolation.
You can first visualize and then save the plot by running
  `python viskit/frontend.py data/part2_c/` and then splitting the figures by `env` and splitting the series by `mode`.


### Question (d): Implementation of look ahead policy
ou will need to fill the code in `part2/lookaheadpolicy.py`. You can run this question by typing 
`python part2/run_part2_d.py -p look_ahead -H HORIZON `, where `HORIZON` represents how many time steps into
the future you do the look ahead. The results should be split by environmen: as in the previous question,
 but this time the series should be splitted by `horizon`.


## Part3: Value Iteration & Fuction Approximation
### Question (a): Implement value iteration for continuous state spaces
You will need to fill the code in `part3/continuous\_value\_iteration.py` and `part3/look\_ahead\_policy.py`. The policy
 implemented will maximize the values function over a random set of actions. You will need to do it for continuous and 
 discrete actions. You can run this question by typing 
`python part3/run_part3_a.py `. Even though we don't ask it in the report you can play with the `batch_size` and `learning_rate` for
 training the nerual network value function, and with the number of actions to maximize value function! You can do that
 by running `python part3/run_part3_a.py -bs BATCH_SIZE -lr LEARNING_RATE -a NUM_ACTIONS `.

### Question (b): Implement look ahead cross-entropy
You will need to fill the code in `part3/look\_ahead\_policy.py`. You can run this question with the command
`python part3/run_part3_b.py -H HORIZON `, where `HORIZON` represents how many time steps into
the future you do the look ahead. You can first visualize and then save the plot by running
  `python viskit/frontend.py data/part3_b/`.
  
  
## Part 4: Vectorized Discretization (Extra Credit)
You will need to fill the code in `part4/discretize.py` below the lines `if self.mode == 'nn'`
 and if `self.mode == 'linear` To run this question 
type `python part4/run_part4.py -m nn `. You can modify the number of points used per dimension in the state
anc action space by running
`python part2/run_part4.py -m INTERPOLATION_MODE -s NUM_STATES`,  where `INTERPOLATION_MODE` is either `nn` for 
nearest-neighbor interpolation, or `linear` for n-linear interpolation and `NUM_STATES` is the number of 
points per state dimension.

