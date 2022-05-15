from experiment_utils import run_experiment
import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', type=str, choices=["one", "many"], default="one",
                        help='one: run one experiments, many: run several experiments')
    parser.add_argument('--config_path', type=str,
                        help='if mode=one: path to config, if mode=many: path to folder with configs')
    parser.add_argument('--exp_folder', type=str, default="experiments",
                        help='folder for save experiments')

    args = parser.parse_args()

    if args.mode == "one":
        with open(args.config_path, "r") as f:
            config = json.load(f)
        save_path = os.path.join(args.exp_folder, config["exp_name"])
        run_experiment(config, save_path)
    elif args.mode == "many":
        for config_name in os.listdir(args.config_path):
            config_path = os.path.join(args.config_path, config_name)
            with open(config_path, "r") as f:
                config = json.load(f)
            save_path = os.path.join(args.exp_folder, config["exp_name"])
            run_experiment(config, save_path)
