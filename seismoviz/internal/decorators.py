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
