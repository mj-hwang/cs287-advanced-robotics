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
    from envs import DoubleIntegratorEnv, MountainCarEnv
    from utils.utils import TabularPolicy, TabularValueFun
    from part1.tabular_value_iteration import ValueIteration
    from part2.look_ahead_policy import LookAheadPolicy
    from part2.discretize import Discretize
    envs = [DoubleIntegratorEnv(), MountainCarEnv()]

    for env in envs:
        env_name = env.__class__.__name__
        exp_dir = os.getcwd() + '/data/part2_ab/%s/mode%s_state_discretization%s/' % (env_name,
                                                                                      args.mode,
                                                                                      str(args.state_discretization)
                                                                                      )
        logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'])
        args_dict = vars(args)
        args_dict['env'] = env_name
        json.dump(vars(args), open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True)

        env = Discretize(env,
                         state_discretization=args.state_discretization,
                         action_discretization=args.action_discretization,
                         mode=args.mode
                         )
        value_fun = TabularValueFun(env)
        if args.policy_type == 'tabular':
            policy = TabularPolicy(env)
        elif args.policy_type == 'look_ahead':
            policy = LookAheadPolicy(env, value_fun, args.horizon)
        else:
            raise NotImplementedError
        algo = ValueIteration(env,
                              value_fun,
                              policy,
                              render=render,
                              max_itr=args.max_iter,
                              num_rollouts=1,
                              render_itr=5,
                              log_itr=5)
        algo.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", "-r", action='store_true',
                        help="Vizualize the policy and contours when training")
    parser.add_argument("--state_discretization", "-s", type=int, default=21,
                        help="Number of points per state dimension to discretize")
    parser.add_argument("--action_discretization", "-a", type=int, default=5,
                        help="Number of points per state dimension to discretize")
    parser.add_argument("--mode", "-m", type=str, default='nn', choices=['nn', 'linear'],
                        help="Mode of interpolate between discrete points")
    parser.add_argument("--policy_type", "-p", type=str, default='tabular', choices=['tabular', 'look_ahead'],
                        help='Type of policy to use. Whether to use look ahead policy or tabular')
    parser.add_argument("--horizon", "-H", type=int, default=1,
                        help='Planning horizon for the look ahead policy')
    parser.add_argument("--max_iter", "-i", type=int, default=150,
                        help='Maximum number of iterations for the value iteration algorithm')
    args = parser.parse_args()
    main(args)
