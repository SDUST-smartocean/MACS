import numpy as np
import inspect
import functools


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper

def make_env(args):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # 加载 simple_spread 场景
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # 创建 world（世界）
    world = scenario.make_world()

    # 创建 multiagent 环境
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.info, scenario.done)

    # 设置玩家个数
    args.n_players = env.n  # 包括所有智能体的数量
    args.n_agents = env.n  # 所有智能体都是操控的玩家

    # 设置每个智能体的观察维度
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]

    # 设置每个智能体的动作维度
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args.action_shape = action_shape[:args.n_agents]

    # 设置动作空间的高值和低值
    args.high_action = 1
    args.low_action = -1

    return env, args
