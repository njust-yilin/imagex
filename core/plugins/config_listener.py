from core.utils.plugins.plugin_loader import get_plugins, PluginConfig

from typing import List


class ConfigListener:
    def __init__(self):
        self._subscribers = []
        self._plugins: List[PluginConfig] = []

    def _is_plugins_difference_with(self, data: List[PluginConfig]):
        return tuple(self._plugins) != tuple(data)

    def register(self, subscriber) -> None:
        self._subscribers.append(subscriber)

    def listen(self)->None:
        plugins = get_plugins()
        if self._is_plugins_difference_with(plugins):
            self._plugins = plugins
            for subscriber in self._subscribers:
                subscriber.on_plugin_loaded(self._plugins)


