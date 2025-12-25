from stable_baselines3 import DQN
import os
from stable_baselines3.common.envs import AUVLinkSelectionEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"model_{self.num_timesteps}")
            print(f"[Callback] Saving model to: {save_file}")
            self.model.save(save_file)
        return True

run_str = '-run-1215-LESCL-num10-llm'
ModelPath = "./models/DQN_BPEnv" + run_str
dataPath = "path/to/data"
tensorboard_log_path = "./BPE_tensorboard"

os.makedirs(tensorboard_log_path, exist_ok=True)
os.makedirs("./models/", exist_ok=True)

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

seed = 1024
num_auv = 10

env = AUVLinkSelectionEnv(
    num_auv=num_auv,
    max_steps=10 * 20,
    seed=seed,
    dataPath=dataPath,
    is_train=True
)

env.reset()

model = DQN(
    "MlpPolicy",
    env,
    device="cuda",
    tensorboard_log=tensorboard_log_path,
    verbose=1,

    # ---- DQN-specific parameters ----
    learning_rate=1e-4,
    gamma=0.99,
    buffer_size=200000,
    learning_starts=5000,           # Start learning after collecting 5k experiences
    batch_size=512,
    tau=1.0,                        # Soft update coefficient for target network
    target_update_interval=1000,    # Update target network every 1000 steps
    train_freq=4,                   # Train once every 4 steps
    gradient_steps=1,
    exploration_fraction=0.3,       # Linearly decrease ε during the first 30% of training
    exploration_final_eps=0.05,     # Final ε
    seed=seed
)

save_callback = SaveModelCallback(
    save_freq=100000,
    save_path="./models/" + run_str
)

model.learn(
    total_timesteps=int(10 * 20 * 40000),
    callback=save_callback
)

model.save(ModelPath)
print(f"DQN training completed and model saved to: {ModelPath}")
