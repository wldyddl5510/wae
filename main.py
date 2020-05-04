import os
import json
import argparse

from run_network import run

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    with open('config.json', 'r') as f:
        config = json.load(f)

    for key, item in config['WAE_MMD_GAUSSIANPRIOR_FASHION_MNIST'].items():
        setattr(args, key, item)

    run(args)

if __name__ == "__main__":
    main()
