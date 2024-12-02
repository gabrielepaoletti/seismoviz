from typing import TypeVar, ParamSpec, Callable
import inspect

P = ParamSpec('P')
T = TypeVar('T')


def sync_metadata(cls, method_name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    A decorator to synchronize the docstring and signature of a method 
    from another class.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        source_method = getattr(cls, method_name)
        func.__doc__ = source_method.__doc__
        func.__signature__ = inspect.signature(source_method)
        func.__annotations__ = source_method.__annotations__
        return func
    return decorator
