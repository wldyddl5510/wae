from wae_agent import WaeAgent
from utils import SAMPLE_PRIOR_DIST, SCALE_LIST
from utils import free_params, frozen_params
import torc

class WaeGanAgent(WaeAgent):


    def __init__(self, args, module, env, device, logger = None):
        super().__init__(args, module, env, device, logger)

    def loss_ae(self, input_x, latent_z, reconst_x):
        return self.compute_wasser_ae()

    def compute_wasser_ae(self, input_x, latent_z, reconst_x):
        pass