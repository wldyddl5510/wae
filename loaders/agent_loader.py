from settings import AGENTS

class AgentLoader:


    def __init__(self, args, module, env, device, logger):
        if args.agent not in [*AGENTS]:
            raise NotImplementedError

        AgentClass = AGENTS[args.agent]
        self.agent = AgentClass(args, module, env, device, logger)