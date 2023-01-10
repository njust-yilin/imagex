import inspect
from collections.abc import Sequence
import warnings


class ComponentManager:
    def __init__(self, name=None):
        self._components_dict = {}
        self._name = name
    
    def __len__(self) -> int:
        return len(self._components_dict)
    
    def __repr__(self) -> str:
        name_str = self._name if self._name else self.__class__.__name__
        return f"{name_str}: {list(self._components_dict)}"
    
    def __getitem__(self, key: str):
        if key not in self._components_dict.keys():
            raise KeyError(f"{key} not found in {self}")
        return self._components_dict[key]

    @property
    def components_dict(self):
        return self._components_dict

    @property
    def name(self) -> str:
        return self._name
    
    def _add_single_component(self, component):
        if not (inspect.isclass(component) or inspect.isfunction(component)):
            raise TypeError(f"{component} must be a class or function")
        
        component_name = component.__name__

        if component_name in self._components_dict:
            warnings.warn(f"{component_name} already exists in {self}")
    
        self._components_dict[component_name] = component
    
    def add_component(self, components):
        if isinstance(components, Sequence):
            for component in components:
                self._add_single_component(component)
        else:
            component = components
            self._add_single_component(component)

        return components


MODELS = ComponentManager("models")
BACKBONES = ComponentManager("backbones")
DATASETS = ComponentManager("datasets")
TRANSFORMS = ComponentManager("transforms")
LOSSES = ComponentManager("losses")

ComponentList = [
    MODELS.components_dict,
    BACKBONES.components_dict,
    DATASETS.components_dict,
    TRANSFORMS.components_dict,
    LOSSES.components_dict
]