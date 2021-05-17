# =============================================
# Default code to train the model. Not used locally.
# This code was copied to a python notebook on the compute servers to train the model
# =============================================

import os

import torch

from constants import MODEL_PATH
from tranformerxl import configure, train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_chkp = MODEL_PATH if os.path.exists(MODEL_PATH) else None
path_chkp = None

# Configure model
model = configure()

# Train
train(model, num_epochs=50, num_segments=300)