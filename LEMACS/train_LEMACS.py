from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import warnings  

warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['SUPPRESS_MA_PROMPT'] = '1'
if __name__ == '__main__':
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    runner.run()
    # runner.evaluate(model_dir="new-model5/simple_spread")
