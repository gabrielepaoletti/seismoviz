#----------------------------------------------------------------------------------------
# IMPORTING MODULES
#----------------------------------------------------------------------------------------

import inspect
from functools import wraps

from seismoviz.plotting.cat_plotting import CatalogPlotter

from typing import Any, Callable, Type

#----------------------------------------------------------------------------------------
# DEFINING DECORATORS
#----------------------------------------------------------------------------------------

def sync_signature(method_name: str, cls: Type) -> Callable:
    """
    A decorator that synchronizes the signature of a method with a target method from a specified class.

    Parameters
    ----------
    method_name : str
        The name of the method in the target class whose signature will be synced.

    cls : Type
        The class from which the target method is retrieved.

    Returns
    -------
    Callable
        The decorated method with a signature synchronized with the target method.
    """
    def decorator(method: Callable) -> Callable:
        """
        Decorates the method by synchronizing its signature with the specified target method.

        Parameters
        ----------
        method : Callable
            The original method to be decorated.

        Returns
        -------
        Callable
            The wrapper function that filters keyword arguments and synchronizes the method signature.
        """
        @wraps(method)
        def wrapper(self, *args, **kwargs) -> Any:
            """
            Wrapper function that filters the keyword arguments and calls the target method.

            Parameters
            ----------
            self : Any
                The instance of the class that contains the decorated method.

            *args : Any
                Positional arguments passed to the method.

            **kwargs : Any
                Keyword arguments passed to the method. Only the valid ones for the target method will be passed.

            Returns
            -------
            result : Any
                The result of calling the target method with the filtered keyword arguments.
            """
            plotter_method = getattr(self.plotter, method_name)  # `self.plotter` è dinamico
            sig = inspect.signature(plotter_method)

            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

            return plotter_method(*args, **filtered_kwargs)

        target_method = getattr(cls, method_name)
        wrapper.__signature__ = inspect.signature(target_method)
        
        return wrapper

    return decorator
