from typing import TypeVar, ParamSpec, Callable
import inspect

P = ParamSpec('P')
T = TypeVar('T')


def sync_metadata(cls, method_name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    A decorator to synchronize the docstring and signature of a method 
    from another class.

    Parameters
    ----------
    cls : type
        The class from which to copy the method's docstring and signature.
    method_name : str
        The name of the method to copy.

    Returns
    -------
    function
        The decorator that updates the docstring and signature of the decorated 
        function.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        source_method = getattr(cls, method_name)
        func.__doc__ = source_method.__doc__
        func.__signature__ = inspect.signature(source_method)
        func.__annotations__ = source_method.__annotations__
        return func
    return decorator
