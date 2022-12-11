from dataclasses import dataclass
import json
from typing import List
from loguru import logger


@dataclass(frozen=True)
class PluginConfig:
    name: str
    module: str


def get_plugins(config_path:str)-> List[PluginConfig]:
    with open(config_path) as fd:
        config = json.load(fd)
        plugins = [PluginConfig(plugin['name'], plugin['module']) for plugin in config['plugins']]
        logger.info(plugins)
        return plugins

