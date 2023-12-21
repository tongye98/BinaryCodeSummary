import os
import argparse
from src.train import train
from src.test import test

def main():
    parser = argparse.ArgumentParser("BinaryCodeSummary")
    parser.add_argument("mode", type=str, default="train", choices=["train", "test", "unittest"])
    parser.add_argument("--config", type=str, default="configs/dataset_gcc-7.3.0_arm_32_O1_strip/O1_test3_cszx.yaml")
    parser.add_argument("--gpu", type=str, default='1')
    parser.add_argument("--ckpt", type=str, default=None, help="Model Checkpoint")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.mode == "train":
        train(cfg_file=args.config)

    elif args.mode == "test":
        test(cfg_file=args.config, ckpt_path=args.ckpt)
    
    elif args.mode == "unittest":
        assert False, "To Do."

    else:
        raise ValueError("Unknown mode!")

if __name__ == "__main__":
    main()