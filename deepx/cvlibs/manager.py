import inspect
from collections.abc import Sequence
import warnings


class ComponentManager:
    def __init__(self, name=None) -> None:
        self._components_dict = dict()
        self._name = name

    def __len__(self) -> int:
        return len(self._components_dict)
    
    def __repr__(self) -> str:
        name_str = self._name if self._name else self.__class__.__name__
        return f'{name_str}:{list(self._components_dict.keys())}'
    
    def __getitem__(self, name:str):
        if name in self._components_dict.keys():
            return self._components_dict[name]
    
    @property
    def components_dict(self):
        return self._components_dict

    @property
    def name(self):
        return self._name

    def _add_single_component(self, component):
        if not (inspect.isclass(component) or inspect.isfunction(component)):
            raise TypeError(f"Component must be a class or a function: but got {type(component)}")
        
        component_name = component.__name__
        if component_name in self._components_dict.keys():
            warnings.warn(f"]{component_name} exists already, now update {component}")
        self.components_dict[component_name] = component
    
    def add_component(self, components):
        if isinstance(components, Sequence):
            for component in components:
                self._add_single_component(component)
        else:
            self._add_single_component(components)
        return components


MODELS = ComponentManager("models")
BACKBONES = ComponentManager("backbones")
DATASETS = ComponentManager("datasets")
TRANSFORMS = ComponentManager("transforms")
LOSSES = ComponentManager("losses")


if __name__ == '__main__':
    @MODELS.add_component
    class Test():
        pass
    print(MODELS.components_dict)