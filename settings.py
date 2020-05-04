# Import envs
from envs.toy_envs import s1
from envs.mnist import mnist, fashion_mnist

# Import modules
from modules.linear_autoencoder import LinearEncoder, LinearDecoder
from modules.dc_autoencoder import DcEncoder, DcDecoder

# Import agents
from agents.wae_mmd_agent import WaeMmdAgent
from agents.info_ae_agent import InfoAeAgent


# datasets
TOY_ENVS = {'S1': s1}
ENVS = {'MNIST': mnist, 'FASHION_MNIST': fashion_mnist}

# Modules
MODULES_ENCODER = {'LINEAR_ENCODER': LinearEncoder, 'DC_ENCODER': DcEncoder}
MODULES_DECODER = {'LINEAR_DECODER': LinearDecoder, 'DC_DECODER': DcDecoder}

#Agents
AGENTS = {'WAE_MMD': WaeMmdAgent, "INFO_AE": InfoAeAgent}
