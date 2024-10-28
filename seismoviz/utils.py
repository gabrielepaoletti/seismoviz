import pyproj
import numpy as np

from numpy.typing import ArrayLike


def convert_to_geographical(
        utmx: ArrayLike,
        utmy: ArrayLike,
        zone: int,
        northern: bool,
        units: str,
        ellps: str = 'WGS84',
        datum: str = 'WGS84'
) -> tuple[ArrayLike, ArrayLike]:
    """
    Converts UTM coordinates to geographical (longitude and latitude)
    coordinates.

    Parameters
    ----------
    utmx : ArrayLike
        Represents the UTM x coordinate(s) (easting).

    utmy : ArrayLike
        Represents the UTM y coordinate(s) (northing).

    zone : int
        The UTM zone number (ranges from 1 to 60).

    northern : bool
        True if in the Northern Hemisphere, False otherwise.

    units : str
        Specifies the units ('m' for meters, 'km' for kilometers).

    ellps : str, optional
        The ellipsoid model for the Earth (default: 'WGS84').

    datum : str, optional
        The geodetic datum (default: 'WGS84').

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        Longitude and latitude values in the same format as the input.
    """
    utm_crs = pyproj.CRS(
        f'+proj=utm +zone={zone} +{"+north" if northern else "+south"} '
        f'+ellps={ellps} +datum={datum} +units={units}'
    )
    geodetic_crs = pyproj.CRS('epsg:4326')

    transformer = pyproj.Transformer.from_crs(
        utm_crs, geodetic_crs, always_xy=True
    )
    lon, lat = transformer.transform(utmx, utmy)
    return lon, lat


def convert_to_utm(
        lon: ArrayLike,
        lat: ArrayLike,
        zone: int,
        units: str,
        ellps: str = 'WGS84',
        datum: str = 'WGS84'
) -> tuple[ArrayLike, ArrayLike]:
    """
    Converts geographical (longitude and latitude) coordinates to UTM
    coordinates.

    Parameters
    ----------
    lon : ArrayLike
        The longitude value(s).
    
    lat : ArrayLike
        The latitude value(s).

    zone : int
        The UTM zone number (ranges from 1 to 60).

    units : str
        Specifies the units ('m' for meters, 'km' for kilometers).

    ellps : str, optional
        The ellipsoid model for the Earth (default: 'WGS84').

    datum : str, optional
        The geodetic datum (default: 'WGS84').

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        UTM x and y coordinates (easting and northing).
    """
    utm_converter = pyproj.Proj(
        proj='utm', zone=zone, units=units, ellps=ellps, datum=datum
    )
    utmx, utmy = utm_converter(np.array(lon), np.array(lat))
    return utmx, utmy
