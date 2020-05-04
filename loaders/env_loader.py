from settings import TOY_ENVS, ENVS

class EnvLoader:


    def __init__(self, args, logger = None):
        self.env_name = args.env
        if (self.env_name not in [*TOY_ENVS]) and (self.env_name not in [*ENVS]):
            raise NotImplementedError

        if self.env_name in [*TOY_ENVS]:
            loader = TOY_ENVS[env_name](args.num_data, args.train_ratio, args.batch_size, args.num_workers)

        elif self.env_name in [*ENVS]:
            image_shape = (args.n_channels, args.img_size, args.img_size)
            loader = ENVS[self.env_name](args.batch_size, args.num_workers, image_shape, args.download_dataset)

        for key, value in loader.items():
            setattr(self, key, value)

        if logger is not None:
            self.logger = logger
            self.logger.info("Sucessfully loaded env: " + self.env_name)