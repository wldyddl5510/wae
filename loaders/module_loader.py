from settings import MODULES_ENCODER, MODULES_DECODER

class ModuleLoader:


    def __init__(self, args, device, logger = None):

        if args.encoder not in [*MODULES_ENCODER]:
            raise NotImplementedError
        if args.decoder not in [*MODULES_DECODER]:
            raise NotImplementedError

        self.encoder = MODULES_ENCODER[args.encoder]

        self.decoder = MODULES_DECODER[args.decoder]
        # self.decoder = DecoderClass(args, device, logger)
