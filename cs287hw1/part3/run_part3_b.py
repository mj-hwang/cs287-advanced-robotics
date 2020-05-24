import logger
import argparse
import os
import json
import numpy as np; np.random.seed(0)


def main(args):
    render = args.render
    if not render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    from envs import DoubleIntegratorEnv, MountainCarEnv, CartPoleEnv, SwingUpEnv
    from utils.utils import VectorizeMujocoEnv
    from part3.look_ahead_policy import LookAheadPolicy
    from utils.value_functions import MLPValueFun
    from part3.continous_value_iteration import ContinousStateValueIteration
    envs = [DoubleIntegratorEnv(), MountainCarEnv(), CartPoleEnv(), SwingUpEnv()]

    for env in envs:
        env_name = env.__class__.__name__
        exp_dir = os.getcwd() + '/data/part3_b/%s/horizon%s' % (env_name, args.horizon)
        logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'])
        args_dict = vars(args)
        args_dict['env'] = env_name
        json.dump(vars(args), open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True)

        value_fun = MLPValueFun(env, hidden_sizes=(512, 512, 512))
        policy = LookAheadPolicy(env,
                                 value_fun,
                                 horizon=args.horizon,
                                 look_ahead_type=args.policy_type,
                                 num_acts=args.num_acts)
        algo = ContinousStateValueIteration(env,
                                            value_fun,
                                            policy,
                                            learning_rate=args.learning_rate,
                                            batch_size=args.batch_size,
                                            num_acts=args.num_acts,
                                            render=args.render,
                                            max_itr=args.max_iter,
                                            log_itr=10)
        algo.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", "-r", action='store_true',
                        help="Vizualize the policy and contours when training")
    parser.add_argument("--policy_type", "-p", type=str, default='cem', choices=['cem', 'rs'],
                        help='Type of policy to use. Whether to use look ahead with cross-entropy \
                        method or random shooting')
    parser.add_argument("--horizon", "-H", type=int, default=1,
                        help='Planning horizon for the look ahead policy')
    parser.add_argument("--max_iter", "-i", type=int, default=250,
                        help='Maximum number of iterations for the value iteration algorithm')
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3,
                        help='Learning rate for training the value function')
    parser.add_argument("--batch_size", "-bs", type=int, default=256,
                        help='batch size for training the value function')
    parser.add_argument("--num_acts", "-a", type=int, default=32,
                        help='Number of actions sampled for maximizing the value function')
    args = parser.parse_args()
    main(args)