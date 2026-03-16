# src/neuralcore/__init__.py
import pkgutil
import importlib
import inspect

__all__ = []

# Iterate over all submodules in the package
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    full_name = f"{__name__}.{module_name}"
    module = importlib.import_module(full_name)
    
    # Add submodule itself
    globals()[module_name] = module
    __all__.append(module_name)
    
    # Add all classes and functions from the module
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            globals()[name] = obj
            __all__.append(name)