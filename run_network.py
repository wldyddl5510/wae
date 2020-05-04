import logging
import logging.config
import json

import torch
import numpy as np

# Load loaders
from loaders.agent_loader import AgentLoader
from loaders.env_loader import EnvLoader
from loaders.module_loader import ModuleLoader

import pdb

def run(args):

    if args.logger:
        with open('logger.json', 'r') as f:
            logger_config = json.load(f)
            logging.config.dictConfig(logger_config)

            # Assign loggers
            root_logger = logging.getLogger("ROOT")
            env_logger = logging.getLogger("ENV")
            module_logger = logging.getLogger("MODULE")
            agent_logger = logging.getLogger("AGENT")
    else:
        root_logger = None
        env_logger = None
        module_logger = None
        agent_logger = None

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set device
    device_option = 'cuda' if args.cuda else 'CPU'
    device = torch.device(device_option)
    if args.logger:
        root_logger.info(device_option + " is activated")

    # Load
    env = EnvLoader(args, logger = env_logger)
    module = ModuleLoader(args, device, logger = module_logger)
    agent = AgentLoader(args, module, env, device, logger = agent_logger).agent

    agent.train()