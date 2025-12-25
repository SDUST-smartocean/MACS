import argparse

"""
Here are the param for the training

"""


def  get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=200, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./new-model5", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluate
    parser.add_argument("--eval_episodes", type=int, default=1, help="number of episodes for evaluating")
    parser.add_argument("--evaluate_len", type=int, default=300, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=True, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    args = parser.parse_args()
    return args
