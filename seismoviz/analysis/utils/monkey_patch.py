# monkey_patch_maps.py
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxesSubplot

# Scegli il default transform per le mappe (ad es. PlateCarree)
DEFAULT_TRANSFORM = ccrs.PlateCarree()

def auto_transform_scatter(self, *args, **kwargs):
    if "transform" not in kwargs:
        kwargs["transform"] = DEFAULT_TRANSFORM
    return self._orig_scatter(*args, **kwargs)

def auto_transform_text(self, *args, **kwargs):
    if "transform" not in kwargs or kwargs["transform"] is None:
        kwargs["transform"] = DEFAULT_TRANSFORM
    return self._orig_text(*args, **kwargs)

def auto_transform_annotate(self, *args, **kwargs):
    if "transform" not in kwargs:
        kwargs["transform"] = DEFAULT_TRANSFORM
    return self._orig_annotate(*args, **kwargs)

# Applica il patch solo se non già patchato
if not hasattr(GeoAxesSubplot, '_orig_scatter'):
    GeoAxesSubplot._orig_scatter = GeoAxesSubplot.scatter
    GeoAxesSubplot.scatter = auto_transform_scatter

if not hasattr(GeoAxesSubplot, '_orig_text'):
    GeoAxesSubplot._orig_text = GeoAxesSubplot.text
    GeoAxesSubplot.text = auto_transform_text

if not hasattr(GeoAxesSubplot, '_orig_annotate'):
    GeoAxesSubplot._orig_annotate = GeoAxesSubplot.annotate
    GeoAxesSubplot.annotate = auto_transform_annotate
