import pandas as pd
import holoviews as hv

from seismoviz.components.catalog import Catalog, SubCatalog
from seismoviz.components.cross_section import CrossSection
from seismoviz.internal.selector import CatalogSelector, CrossSectionSelector


def read_catalog(path: str, **kwargs) -> Catalog:
    """
    Reads a CSV file and returns a Catalog object.

    .. warning::
        The input CSV file must contain the following columns: 
        ``lon``, ``lat``, ``time``, ``depth``, ``mag``, and ``id``.
        If any of these columns are missing, an error will be raised.

    Parameters
    ----------
    path : str
        The path to the CSV file containing the seismic catalog.

    **kwargs
        Additional keyword arguments to pass to `pandas.read_csv`.
    
    Returns
    -------
    Catalog
        An instance of the ``Catalog`` class with the data loaded.

    Example
    -------

    Basic usage:
    
    .. code-block:: python

        # Reading a catalog with default settings
        catalog = read_catalog(
            path='seismic_data.csv'
        )

    For a more customized behavior, you can pass ``pd.read_csv()`` arguments:

    .. code-block:: python

        # Reading a catalog with a custom delimiter and selected columns
        catalog = read_catalog(
            path='seismic_data.csv', 
            delimiter=';', 
            usecols=['id', 'lon', 'lat', 'time', 'mag']
        )
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
    Creates a seismic cross-section from a given seismic catalog.

    Parameters
    ----------
    catalog : Catalog
        An instance of the `Catalog` class containing seismic event data.
    
    center : tuple[float, float]
        A tuple representing the geographical coordinates (longitude, latitude) 
        of the center of the cross-section.

    num_sections : tuple[int, int]
        A tuple specifying the number of sections to create to the left and 
        right of the center (e.g., (2, 2) will create 2 sections on each side 
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
        An instance of the CrossSection class with the seismic events that fit 
        the specified parameters.
    """
    return CrossSection(
        catalog, center, num_sections, thickness, strike, 
        map_length, depth_range, section_distance
    )


class select_on_map:
    """
    Simulates a function for selecting data from a map, using 
    a CatalogSelector to interact with seismic data.

    Parameters
    ----------
    catalog : Catalog
        The Catalog object containing seismic data and plotting 
        configurations.

    size : float, optional
        The size of the points in the scatter plot. Default is 1.

    color : str, optional
        The color of the points in the scatter plot. Default is 'black'.
    """

    def __init__(
        self,
        catalog: Catalog,
        size: float = 1,
        color: str = 'black'
    ) -> None:
        self._selector = CatalogSelector(catalog)
        self._selector.select(size=size, color=color)

    def confirm_selection(self) -> SubCatalog:
        """
        Confirms the selection and returns a SubCatalog of the selected data.

        Returns
        -------
        SubCatalog
            A SubCatalog object containing the selected data from the 
            cross-section.
        """
        return SubCatalog(
            data=self._selector.sd,
            selected_from='Catalog'
        )


class select_on_section:
    """
    Simulates a function for selecting data from a cross-section, using 
    a CrossSectionSelector to interact with seismic data.

    Parameters
    ----------
    cross_section : CrossSection
        The CrossSection object containing seismic data and plotting 
        configurations.

    size : float, optional
        The size of the points in the scatter plot. Default is 1.

    color : str, optional
        The color of the points in the scatter plot. Default is 'black'.
    """

    def __init__(
        self,
        cross_section: CrossSection,
        size: float = 1,
        color: str = 'black'
    ) -> None:
        self._selector = CrossSectionSelector(cross_section)
        self._selector.select(size=size, color=color)

    def confirm_selection(self) -> SubCatalog:
        """
        Confirms the selection and returns a SubCatalog of the selected data.

        Returns
        -------
        SubCatalog
            A SubCatalog object containing the selected data from the 
            cross-section.
        """
        return SubCatalog(
            data=self._selector.sd,
            selected_from='CrossSection'
        )