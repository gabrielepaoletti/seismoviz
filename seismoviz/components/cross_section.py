import numpy as np
import pandas as pd

from seismoviz.components import Catalog
from seismoviz.utils import convert_to_utm
from seismoviz.components.analysis import Analyzer
from seismoviz.internal.decorators import sync_metadata
from seismoviz.components.visualization import CrossSectionPlotter
from seismoviz.internal.mixins import DunderMethodMixin, GeospatialMixin

from numpy.typing import ArrayLike


class CrossSection(GeospatialMixin, DunderMethodMixin):
    """
    Represents a seismic cross section.

    Attributes
    ----------
    data : pd.DataFrame
        A DataFrame containing the seismic event data for each cross-sectional slice.

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

    Raises
    ------
    ValueError
        If the provided catalog is not an instance of the `Catalog` class.
    """
    def __init__(
            self, 
            data: Catalog, 
            center: tuple[float, float], 
            num_sections: tuple[int, int], 
            tickness: int, 
            strike: int,
            map_length: int, 
            depth_range: tuple[float, float], 
            section_distance: int = 0
    ) -> None:
        if isinstance(data, Catalog):
            self.catalog = data
            self.data = self.catalog.data
        else:
            raise ValueError('The input must be a Catalog object.')

        super().__init__()

        self.center = center
        self.num_sections = num_sections
        self.tickness = tickness
        self.strike = strike
        self.map_length = map_length
        self.depth_range = depth_range
        self.section_distance = section_distance
        self.data = self._cross_section()

        self._plotter = CrossSectionPlotter(self)
        self._analyzer = Analyzer(self)

    @sync_metadata(Analyzer, 'filter')
    def filter(self, **kwargs):
        return self._analyzer.filter(**kwargs)

    @sync_metadata(Analyzer, 'sort')
    def sort(self, **kwargs):
        return self._analyzer.sort(**kwargs)

    @sync_metadata(Analyzer, 'deduplicate_events')
    def deduplicate_events(self):
        return self._analyzer.deduplicate_events()

    @sync_metadata(CrossSectionPlotter, 'plot_sections')
    def plot_sections(self, **kwargs):
        self._plotter.plot_sections(**kwargs)

    @sync_metadata(CrossSectionPlotter, 'plot_section_lines')
    def plot_section_lines(self, **kwargs):
        self._plotter.plot_section_lines(**kwargs)
    
    @sync_metadata(CrossSectionPlotter, 'plot_section_lines')
    def plot_magnitude_time(self, **kwargs):
        self._plotter.plot_magnitude_time(**kwargs) 

    @sync_metadata(CrossSectionPlotter, 'plot_event_timeline')
    def plot_event_timeline(self, **kwargs) -> None:
        self._plotter.plot_event_timeline(**kwargs)

    @sync_metadata(CrossSectionPlotter, 'plot_attribute_distributions')
    def plot_attribute_distributions(self, **kwargs) -> None:
        self._plotter.plot_attribute_distributions(**kwargs)

    @sync_metadata(Analyzer, 'fmd')
    def fmd(self, **kwargs):
        self._analyzer.fmd(**kwargs)
    
    @sync_metadata(Analyzer, 'estimate_b_value')
    def estimate_b_value(self, bin_size: float, mc: str | float, **kwargs):
        if mc == 'maxc':
            mc_maxc = self._analyzer._maxc(bin_size=bin_size)
            return self._analyzer.estimate_b_value(
                bin_size=bin_size, mc=mc_maxc, **kwargs
            )
        elif isinstance(mc, int) or isinstance(mc, float):
            return self._analyzer.estimate_b_value(
                bin_size=bin_size, mc=mc, **kwargs
            )
        else:
            raise ValueError('Mc value is not valid.')

    @sync_metadata(Analyzer, 'interevent_time')
    def interevent_time(self, **kwargs):
        self._analyzer.interevent_time(**kwargs)

    @sync_metadata(Analyzer, 'cov')
    def cov(self, **kwargs):
        self._analyzer.cov(**kwargs)

    @staticmethod
    def _distance_point_from_plane(
            x: float, 
            y: float, 
            z: float, 
            normal: ArrayLike, 
            origin: ArrayLike
    ) -> float:
        """
        Calculate the perpendicular distance of a point (x, y, z) from 
        a plane defined by its normal vector and origin.
        """
        d = -normal[0] * origin[0] - normal[1] * origin[1] - normal[2] * origin[2]
        dist = np.abs(normal[0] * x + normal[1] * y + normal[2] * z + d)
        dist /= np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        return dist

    @staticmethod
    def _section_center_positions(
            center_x: float, 
            center_y: float, 
            section_centers: float, 
            strike: float
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculate the positions of section centers based on a reference 
        point and strike angle.
        """
        angle_rad = np.pi / 2 - np.radians(strike)
        return (
            center_x + section_centers * np.cos(angle_rad), 
            center_y + section_centers * np.sin(angle_rad)
        )

    def _cross_section(self) -> pd.DataFrame:
        """
        Generate cross-sectional slices of seismic event data.
        """
        self.data.depth = np.abs(self.data.depth)

        self._utmx, self._utmy = convert_to_utm(
            self.data.lon, self.data.lat, zone=self.zone, units='km', 
            ellps='WGS84', datum='WGS84'
        )
        center_utmx, center_utmy = convert_to_utm(
            self.center[0], self.center[1], zone=self.zone, units='km', 
            ellps='WGS84', datum='WGS84'
        )

        normal_tostrike = self.strike - 90
        normal_ref = [
            np.cos(normal_tostrike * np.pi / 180), 
            -np.sin(normal_tostrike * np.pi / 180), 0
        ]

        centers_distro = np.arange(
            -self.num_sections[0] * self.section_distance, 
            self.num_sections[1] * self.section_distance + 1, 
            self.section_distance
        )
        centers_depths = -10 * np.ones(len(centers_distro))
        center_xs, center_ys = self._section_center_positions(
            center_utmx, center_utmy, centers_distro, self.strike
        )
        self._center_coords = np.array([center_xs, center_ys, centers_depths]).T

        section_dataframes = []

        for section in range(len(centers_distro)):
            dist = self._distance_point_from_plane(
                self._utmx, self._utmy, -self.data['depth'], normal_ref, 
                self._center_coords[section]
            )
            in_depth_range = (
                (self.data['depth'] >= self.depth_range[0]) & 
                (self.data['depth'] <= self.depth_range[1])
            )
            on_section_coords = (
                (self._utmy - self._center_coords[section][1]) * normal_ref[0] - 
                (self._utmx - self._center_coords[section][0]) * normal_ref[1]
            )

            close_and_in_depth = np.where(
                (dist < self.tickness) & in_depth_range & 
                (np.abs(on_section_coords) < self.map_length / 2)
            )

            section_df = self.data.iloc[close_and_in_depth].copy()
            section_df['on_section_coords'] = on_section_coords[close_and_in_depth]
            section_df['section_id'] = section
            section_dataframes.append(section_df)

        section_dataframes = pd.concat(
            section_dataframes, 
            keys=[df['section_id'].iloc[0] for df in section_dataframes], 
            names=['section_id']
        )

        return section_dataframes