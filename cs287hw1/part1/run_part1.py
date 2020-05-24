import argparse
import os
import logger
import json


def main(args):
    render = args.render
    if not render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    from utils.utils import TabularPolicy, TabularValueFun
    from part1.tabular_value_iteration import ValueIteration
    from envs import Grid1DEnv, GridWorldEnv
    envs = [GridWorldEnv(seed=0), GridWorldEnv(seed=1)]

    for env in envs:
        env_name = env.__name__
        exp_dir = os.getcwd() + '/data/part1/%s/policy_type%s_temperature%s/' % (env_name, args.policy_type, args.temperature)
        logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'])
        args_dict = vars(args)
        args_dict['env'] = env_name
        json.dump(vars(args), open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True)

        policy = TabularPolicy(env)
        value_fun = TabularValueFun(env)
        algo = ValueIteration(env,
                              value_fun,
                              policy,
                              policy_type=args.policy_type,
                              render=render,
                              temperature=args.temperature)
        algo.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_type", "-p", type=str, default='deterministic', choices=["deterministic", "max_ent"],
                        help="Whether to train a deterministic policy or a maximum entropy one")
    parser.add_argument("--render", "-r", action='store_true', help="Vizualize the policy and contours when training")
    parser.add_argument("--temperature", "-t", type=float, default=1.,
                        help="Temperature parameter for maximum entropy policies")
    args = parser.parse_args()
    main(args)
