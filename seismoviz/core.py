import pandas as pd

from seismoviz.catalog import Catalog
from seismoviz.cross_section import CrossSection


def read_catalog(path: str) -> Catalog:
    """
    Reads a CSV file and returns a Catalog object.
    
    Parameters
    ----------
    path : str
        The path to the CSV file containing the seismic catalog.
    
    Returns
    -------
    Catalog
        An instance of the Catalog class with the data loaded.
    """
    data = pd.read_csv(path)
    data.time = pd.to_datetime(data.time)
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
