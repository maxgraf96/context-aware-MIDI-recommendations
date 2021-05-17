# =============================================
# Contains all constants used in the system
# as well as a dynamic model path
# =============================================

import os

calling_dir = os.getcwd()
prefix = '' if 'transformer' in calling_dir else 'transformer/'
MODEL_PATH = prefix + 'REMI-my-checkpoint/model.pt'

INPUT_LENGTH = 256

# Markov approach
HIST_DEPTH = 15
N_TOKENS = 308
N_TAKES = 5