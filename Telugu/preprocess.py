import argparse
import yaml

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastSpeech 2 Feature Extraction & Preprocessing")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to unified config.yaml",
    )
    parser.add_argument(
        "pos_config",
        nargs="?",
        default=None,
        help="Optional positional path to config.yaml",
    )
    args = parser.parse_args()

    config_path = args.pos_config if args.pos_config is not None else args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
