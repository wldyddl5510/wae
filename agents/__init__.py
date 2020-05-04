import sys
import os
from paths import PROJECT_PATH

sys.path.append("../")
sys.path.append(os.path.join(PROJECT_PATH, 'agents'))

from autoencoder_agent import *
from wae_agent import *
from wae_mmd_agent import *
from info_ae_agent import *
from info_wae_mmd_ae_agent import *