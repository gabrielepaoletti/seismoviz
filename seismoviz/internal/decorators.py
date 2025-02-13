import functools
import inspect

import inspect
from typing import Type

def sync_methods(sources: list[Type]) -> callable:
    """
    Returns a decorator to sync the docstring and signature of methods from the
    sources to the target class if method names match.
    """
    def decorator(cls: Type) -> Type:
        """
        Decorates a class to update method docstrings and signatures based on 
        the provided source classes.
        """
        for source_cls in sources:
            for name, source_method in inspect.getmembers(
                source_cls, predicate=inspect.isfunction
            ):
                if hasattr(cls, name):
                    target_method = getattr(cls, name)

                    if source_method.__doc__:
                        target_method.__doc__ = source_method.__doc__
                    
                    target_method.__signature__ = inspect.signature(source_method)
        return cls
    return decorator