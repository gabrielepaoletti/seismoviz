import pandas as pd

from seismoviz.components import (
    Catalog, CrossSection
)
from seismoviz.internal.selector import (
    MapSelection, CrossSectionSelection, CustomSelection
)


def read_catalog(path: str, **kwargs) -> Catalog:
    """
    Reads a CSV file and returns a ``Catalog`` object.

    Parameters
    ----------
    path : str
        The path to the CSV file containing the seismic catalog.

    **kwargs
        Additional keyword arguments to pass to ``pandas.read_csv()``.
    
    Returns
    -------
    Catalog
        An instance of the ``Catalog`` class with the data loaded.

    Examples
    --------

    Basic usage:
    
    .. code-block:: python

        # Reading a catalog with default settings
        catalog = sv.read_catalog(
            path='seismic_data.csv'
        )

    For a more customized behavior, you can pass ``pd.read_csv()`` arguments:

    .. code-block:: python

        # Reading a catalog with a custom delimiter and selected columns
        catalog = sv.read_catalog(
            path='seismic_data.csv', 
            delimiter=';', 
            usecols=['id', 'lon', 'lat', 'depth', 'time', 'mag']
        )
    
    .. warning::
        The input CSV file must contain the following columns: 
        ``lon``, ``lat``, ``time``, ``depth``, ``mag``, and ``id``.
        If any of these columns are missing, an error will be raised.
    """
    data = pd.read_csv(path, parse_dates=['time'], **kwargs)
    return Catalog(data)


def create_cross_section(
    catalog: Catalog, 
    center: tuple[float, float], 
    num_sections: tuple[int, int], 
    thickness: int, 
    strike: int,
    map_length: float, 
    depth_range: tuple[float, float], 
    section_distance: float = 1.0
) -> CrossSection:
    """
    Creates a seismic cross-section from a given ``Catalog``.

    Parameters
    ----------
    catalog : Catalog
        An instance of the ``Catalog`` class containing seismic event data.
    
    center : tuple[float, float]
        A tuple representing the geographical coordinates (longitude, latitude) 
        of the center of the cross-section.

    num_sections : tuple[int, int]
        A tuple specifying the number of sections to create to the left and 
        right of the center (e.g., ``(2, 2)`` will create 2 sections on each side 
        of the center).
    
    thickness : int
        The maximum distance (in km) that events can be from the cross-section 
        plane to be included in the section.
    
    strike : int
        The strike angle (in degrees) of the cross-section, measured clockwise 
        from north. Cross section will be computed perpendicular to strike.
    
    map_length : float
        The length of the cross-section (in km), which determines the horizontal 
        extent of the plotted data.
    
    depth_range : tuple[float, float]
        A tuple specifying the minimum and maximum depth (in km) of events to 
        include in the cross-section.

    section_distance : float, optional
        The distance (in km) between adjacent sections. Default is 1.

    Returns
    -------
    CrossSection
        An instance of the ``CrossSection`` class with the seismic events that fit 
        the specified parameters.

    Examples
    --------
    .. code-block:: python

        cs = sv.create_cross_section(
            catalog=catalog,
            center=(13.12, 42.83),
            num_sections=(2,2),
            thickness=1,
            strike=155,
            map_length=40,
            depth_range=(0, 10),
            section_distance=2
        )

    The output will be a ``CrossSection`` object. To access the data, you can 
    use the ``cs.data`` attribute, which is a DataFrame containing all the events 
    within the sections. Each event is labeled with a ``section_id``, allowing 
    you to easily identify which section it belongs to.
    """
    return CrossSection(
        catalog, center, num_sections, thickness, strike, 
        map_length, depth_range, section_distance
    )


def select(
        instance: type,
        x: str = None,
        y: str = None,
        custom: bool = False,
        section_ids: list = None,
        **kwargs
    ):
    """
    Launch an interactive tool to select seismic events.

    .. note::
        When ``custom`` is ``True``, you can choose which graph to use for the 
        sections by selecting the columns with ``x`` and ``y``. Otherwise, the 
        selection  is set automatically: a ``Catalog`` instance produces a map 
        selection, while a ``CrossSection`` instance produces a cross-section 
        selection.

    Parameters
    ----------
    instance : type
        An object containing seismic data and plotting configurations.

    x : str, optional
        The column name for the x-axis variable. Required if ``custom`` is ``True``.

    y : str, optional
        The column name for the y-axis variable. Required if ``custom`` is ``True``.

    custom : bool, optional
        Flag to indicate if a custom selection should be used. Defaults to ``False``.
        
    section_ids : int, list, optional
        When the instance is a ``CrossSection``, this parameter allows specifying
        which section(s) to display. Can be a single section ID (int) or a list of IDs.
        If specified, only events from the selected section(s) will be displayed.

    **kwargs : dict, optional
        Additional keyword arguments for customizing the selection.

    Returns
    -------
    An interactive selection object with a ``confirm_selection()`` method.

    Raises
    ------
    ValueError
        If ``custom`` is True and either ``x`` or ``y`` is not provided, or if
        the type of ``instance`` is unsupported.
    """
    if custom:
        if x is None or y is None:
            raise ValueError('x and y must be provided if custom is True.')
        
        if not isinstance(instance, (Catalog, CrossSection)):
            raise ValueError('Unsupported type for instance.') 
        
        return CustomSelection(instance, x, y, **kwargs)

    elif isinstance(instance, Catalog):
        return MapSelection(instance, **kwargs)
    
    elif isinstance(instance, CrossSection):
        return CrossSectionSelection(instance, section_ids=section_ids, **kwargs)

    else:
        raise ValueError('Unsupported type for instance.')