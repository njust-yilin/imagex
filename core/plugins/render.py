from imagex.utils.import_helper import load_module
from imagex.utils.plugins.config_listener import ConfigListener

class Renderer:
    def __init__(self, listener: ConfigListener) -> None:
        self._listener = listener
        if listener:
            listener.register(self)
        self._plugins = None
    
    def on_plugin_loaded(self, plugins):
        self._plugins = plugins
        if self._plugins:
            self.render()

    def render(self) -> None:
        print(self.plugins)
    


