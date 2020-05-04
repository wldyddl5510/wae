import sys
import os
from paths import PROJECT_PATH

sys.path.append("../")
sys.path.append(os.path.join(PROJECT_PATH, 'modules'))

from autoencoder import *
from dc_autoencoder import *
from linear_autoencoder import *