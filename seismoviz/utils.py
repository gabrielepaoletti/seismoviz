#----------------------------------------------------------------------------------------
# IMPORTING MODULES
#----------------------------------------------------------------------------------------

import pyproj

import numpy as np

from numpy.typing import ArrayLike
from typing import Tuple

#----------------------------------------------------------------------------------------
# DEFINING FUNCTIONS
#----------------------------------------------------------------------------------------

def convert_to_geographical(utmx: ArrayLike, utmy: ArrayLike, zone: int, northern: bool, units: str, ellps: str='WGS84',
                            datum: str='WGS84')-> Tuple[ArrayLike, ArrayLike]:
    """
    Converts UTM coordinates to geographical (longitude and latitude) coordinates.

    .. note::
        This function is capable of handling both individual floating-point numbers and bulk data in the form of lists, arrays, or pandas Series for the UTM coordinates.

    Parameters
    ----------
    utmx : ArrayLike
        Represents the UTM x coordinate(s) (easting), indicating the eastward-measured distance from the meridian of the UTM zone.

    utmy : ArrayLike
        Represents the UTM y coordinate(s) (northing), indicating the northward-measured distance from the equator.

    zone : int
        The UTM zone number that the coordinates fall into, which ranges from 1 to 60. This number helps identify the specific longitudinal band used for the UTM projection.

    northern : bool
        A boolean flag that specifies the hemisphere of the coordinates. Set to True if the coordinates are in the Northern Hemisphere, and False if they are in the Southern Hemisphere.

    units : str
        Specifies the units of the input UTM coordinates. Accepted values are 'm' for meters and 'km' for kilometers. This ensures that the conversion process accurately interprets the scale of the input coordinates.

    ellps : str, optional
        The ellipsoid model used for the Earth's shape in the conversion process, with 'WGS84' as the default. The choice of ellipsoid affects the accuracy of the conversion, as it defines the geometric properties of the Earth model used.

    datum : str, optional
        The geodetic datum that specifies the coordinate system and ellipsoidal model of the Earth, for calculating geographical coordinates. The default is 'WGS84', which is a widely used global datum that provides a good balance of accuracy for global applications.

    Returns
    -------
    lon : ArrayLike
        The longitude value(s) obtained from the conversion, presented in the same format as the input coordinates.

    lat : ArrayLike
        The latitude value(s) obtained from the conversion, presented in the same format as the input coordinates.

    See Also
    --------
    convert_to_utm : Converts geographical (longitude and latitude) coordinates to UTM coordinates.

    Examples
    --------
    .. code-block:: python

        import seismutils.geo as sug

        utmx, utmy = 350, 4300  # UTM coordinates
        
        lon, lat = sug.convert_to_geographical(
            utmx=utmx,
            utmy=utmy,
            zone=33,
            northern=True,
            units='km',
        )
        >>> 'Latitude: 13.271772, Longitude: 38.836032'
    """
    utm_crs = pyproj.CRS(f'+proj=utm +zone={zone} +{"+north" if northern else "+south"} +ellps={ellps} +datum={datum} +units={units}')
    geodetic_crs = pyproj.CRS('epsg:4326')
    
    transformer = pyproj.Transformer.from_crs(utm_crs, geodetic_crs, always_xy=True)
    
    lon, lat = transformer.transform(utmx, utmy)
    return lon, lat

def convert_to_utm(lon: ArrayLike, lat: ArrayLike, zone: int, units: str, ellps: str='WGS84', datum: str='WGS84') -> Tuple[ArrayLike, ArrayLike]:
    """
    Converts geographical (longitude and latitude) coordinates to UTM coordinates.

    .. note::
        This function is capable of handling both individual floating-point numbers and bulk data in the form of lists, arrays or pandas Series for the UTM coordinates.

    Parameters
    ----------
    lon : ArrayLike
        The longitude value(s).
    
    lat : ArrayLike
        The latitude value(s)
        
    zone : int
        The UTM zone number that the coordinates fall into, which ranges from 1 to 60. This number helps identify the specific longitudinal band used for the UTM projection.


    units : str
        Specifies the units of the input UTM coordinates. Accepted values are 'm' for meters and 'km' for kilometers. This ensures that the conversion process accurately interprets the scale of the input coordinates.

    ellps : str, optional
        The ellipsoid model used for the Earth's shape in the conversion process, with 'WGS84' as the default. The choice of ellipsoid affects the accuracy of the conversion, as it defines the geometric properties of the Earth model used.

    datum : str, optional
        The geodetic datum that specifies the coordinate system and ellipsoidal model of the Earth, for calculating geographical coordinates. The default is 'WGS84', which is a widely used global datum that provides a good balance of accuracy for global applications.

    Returns
    -------
    utmx : ArrayLike
        The resulting UTM x coordinates (easting), indicating the distance eastward from the central meridian of the UTM zone, presented in the same format as the input coordinates.
        
    utmy : ArrayLike
        The resulting UTM y coordinates (northing), indicating the distance northward from the equator, presented in the same format as the input coordinates.

    See Also
    --------
    convert_to_geographical : Converts UTM coordinates to geographical (longitude and latitude) coordinates.

    Examples
    --------
    .. code-block:: python

        import seismutils.geo as sug

        lon, lat = 13.271772, 38.836032  # Geographical coordinates

        utmx, utmy = sug.convert_to_utm(
            lon=lon,
            lat=lat,
            zone=33,
            units='km'
        )

        print(f'UTM X: {utmx}, UTM Y: {utmy}')
        >>> 'UTM X: 350, UTM Y: 4300'
    """
    utm_converter = pyproj.Proj(proj='utm', zone=zone, units=units, ellps=ellps, datum=datum)
    utmx, utmy = utm_converter(np.array(lon), np.array(lat))
    return utmx, utmy