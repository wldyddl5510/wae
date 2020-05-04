from autoencoder_agent import AutoencoderAgent
from utils import ASSYMETRIC_DIVERGENCES, srw

class InfoAeFoldedAgent(AutoencoderAgent):


    def __init__(self, args, module, env, device, logger = None):
        super().__init__(args, module, env, device, logger)

        # KL of X from X_tilda
        self.lambda_input_reconst = args.lambda_input_reconst
        # KL of X_tilda from X
        self.lambda_reconst_input = args.lambda_reconst_input
        # Subspace robust wasserstein
        self.lambda_subspace_wasser = args.lambda_subspace_wasser
        # divergence to use
        self.assym_div = ASSYMETRIC_DIVERGENCE[args.assym_div]

    def loss_ae(self, input_x, latent_z, reconst_x):
        div_input_reconst = self.assym_div(input_x, reconst_x, self.device)
        div_reconst_input = self.assym_div(reconst_x, input_x, self.device)
        subspace_wasser = srw(input_x, reconst_x, self.dim_z, self.device)

        # L = KL(X | X~) - KL(X~ | X) + W(X, X~)
        loss = self.lambda_input_reconst * div_input_reconst - self.lambda_reconst_input * div_reconst_input + self.lambda_wasser_subspace * subspace_wasser
        return loss
