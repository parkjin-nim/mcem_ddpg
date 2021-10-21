from CEM_64x64 import *
from DDPG import *
from CEM_DDPG import *
import os
import numpy as np
from Utils import mplotLearning

Mountain_log_files = ["TD3_64x64_BipedalWalker-v3_1000_games.npy",
                    "TD3_64x64_BipedalWalker-v3_1000_games.npy",
                    "CEM_TD3_64x64_BipedalWalker-v3_1000_games.npy"]

Lunar_log_files = ["TD3_64x64_BipedalWalker-v3_1000_games.npy",
                    "TD3_64x64_BipedalWalker-v3_1000_games.npy",
                    "CEM_TD3_64x64_BipedalWalker-v3_1000_games.npy"]

Walker_log_files = ["TD3_64x64_BipedalWalker-v3_1000_games.npy",
                    "TD3_64x64_BipedalWalker-v3_1000_games.npy",
                    "CEM_TD3_64x64_BipedalWalker-v3_1000_games.npy"]

Cheetah_log_files = ["TD3_64x64_BipedalWalker-v3_1000_games.npy",
                    "TD3_64x64_BipedalWalker-v3_1000_games.npy",
                    "CEM_TD3_64x64_BipedalWalker-v3_1000_games.npy"]

log_dir = "./npy/"

if __name__ == "__main__":
    log_arr = []
    for f in Walker_log_files:
        file = os.path.join(log_dir, f)
        log = np.load(file)
        log_arr.append(log)

    log_arr = np.array(log_arr)
    mplotLearning(log_arr, "./charts/Walker")
