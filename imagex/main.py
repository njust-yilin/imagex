from loguru import logger
import json
import time
from hashlib import sha256
import sys
from typing import List, Dict

from settings import configs
from core.utils.license.verification_license import rsa_decrypt, get_hardware_id
from core.utils.plugins.plugin_loader import get_plugins, PluginConfig
from core.utils.import_helper import load_module
from services.service import Service


def verify_license():
    try:
        with open(configs.LICENSE_PATH, "rb") as fd:
            content = rsa_decrypt(fd.read())
            license = json.loads(content)
            assert license['author'] == 'yilin'
            assert license['organization'] == 'imagex'
            assert license['license_version'] == '1.0'
            if license['expires'] > 0:
                assert license['expires'] >= time.time()
            hardware_id = get_hardware_id()

            with open('../core/utils/license/verification_license.py', 'rb') as fd:
                content = fd.read()
                sign = sha256(content).hexdigest()
            hardware_id = sha256((hardware_id + sign).encode()).hexdigest()
            # logger.info(hardware_id)
            assert license['hardware_id'] == hardware_id
    except AssertionError:
        logger.error("License Certificate invalid.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"License not found at {configs.LICENSE_PATH}")
        sys.exit(1)
    logger.info("License certificate successfully signed!")

def main():
    verify_license()
    logger.info("Starting Application")

    plugins: List[PluginConfig] = get_plugins(configs.SERVICE_CONFIG_PATH)
    process_map:Dict[str, Service] = {}
    for plugin_config in plugins:
        module = load_module(plugin_config.module)
        service:Service = module.get_service()
        service.start()
        process_map[plugin_config.name] = service

    logger.info("Exiting Application")


if __name__ == "__main__":
    main()
