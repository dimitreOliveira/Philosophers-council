import logging
import pprint

import yaml


def parse_configs(configs_path: str) -> dict:
    """Parse configs from the YAML file.

    Args:
        configs_path (str): Path to the YAML file

    Returns:
        dict: Parsed configs
    """
    configs = yaml.safe_load(open(configs_path, "r"))
    logger.info(f"Configs: {pprint.pformat(configs)}")
    return configs


CONFIGS_PATH = "configs.yaml"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

configs = parse_configs(CONFIGS_PATH)
