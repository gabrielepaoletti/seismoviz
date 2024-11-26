import inspect
from functools import wraps
from collections.abc import Callable


def sync_signature(attribute_name: str, method_name: str) -> Callable:
    """
    A decorator that synchronizes the signature of a method with a target 
    method from a specified attribute's class.

    Parameters
    ----------
    attribute_name : str
        The name of the attribute containing the target method.
    method_name : str
        The name of the method in the target attribute's class whose signature
        will be synced.

    Returns
    -------
    Callable
        The decorated method with a signature synchronized with the target 
        method.
    """
    def decorator(method: Callable) -> Callable:
        """
        Decorates the method by synchronizing its signature with the specified 
        target method.

        Parameters
        ----------
        method : Callable
            The original method to be decorated.

        Returns
        -------
        Callable
            The wrapper function that filters keyword arguments and synchronizes 
            the method signature.
        """
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            """
            Wrapper function that filters the keyword arguments and calls the 
            target method.

            Parameters
            ----------
            self : any type
                The instance of the class that contains the decorated method.
            *args : any type
                Positional arguments passed to the method.
            **kwargs : any type
                Keyword arguments passed to the method. Only the valid ones for 
                the target method will be passed.

            Returns
            -------
            any type
                The result of calling the target method with the filtered keyword 
                arguments.
            """
            target_instance = getattr(self, attribute_name)
            target_method = getattr(target_instance, method_name)
            sig = inspect.signature(target_method)

            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in sig.parameters
            }

            return target_method(*args, **filtered_kwargs)

        def set_signature_on_wrapper(instance):
            target_class = type(getattr(instance, attribute_name))
            target_method = getattr(target_class, method_name)
            wrapper.__signature__ = inspect.signature(target_method)

        wrapper._set_signature_on_wrapper = set_signature_on_wrapper

        return wrapper

    return decorator


from typing import TypeVar, ParamSpec, Callable
import inspect

P = ParamSpec('P')
T = TypeVar('T')

def sync_metadata(cls, method_name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decoratore per sincronizzare la docstring e la firma di un metodo da un'altra classe.

    Args:
        cls: La classe dalla quale copiare il metodo.
        method_name: Il nome del metodo da copiare.

    Returns:
        Il decoratore che aggiorna la funzione decorata.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        source_method = getattr(cls, method_name)
        func.__doc__ = source_method.__doc__
        func.__signature__ = inspect.signature(source_method)
        func.__annotations__ = source_method.__annotations__
        return func
    return decorator
