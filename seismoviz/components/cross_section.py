import numpy as np
import pandas as pd

from seismoviz.utils import convert_to_utm
from seismoviz.components.catalog import Catalog
from seismoviz.internal.decorators import sync_signature
from seismoviz.internal.mixins import DunderMethodMixin, GeospatialMixin
from seismoviz.components.plotters.crs_plotter import CrossSectionPlotter

from numpy.typing import ArrayLike


class CrossSection(GeospatialMixin, DunderMethodMixin):
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
            self.data = data.data
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

        Parameters
        ----------
        x : float
            The x-coordinate of the point.

        y : float
            The y-coordinate of the point.

        z : float
            The z-coordinate of the point.

        normal : ArrayLike
            The normal vector to the plane, defined by [nx, ny, nz].

        origin : ArrayLike
            The origin point on the plane, defined by [ox, oy, oz].

        Returns
        -------
        float
            The perpendicular distance from the point to the plane.
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

        Parameters
        ----------
        center_x : float
            The x-coordinate of the reference center.

        center_y : float
            The y-coordinate of the reference center.

        section_centers : ArrayLike
            The distances of section centers relative to the reference 
            point.

        strike : float
            The strike angle of the sections in degrees, measured 
            clockwise from north.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            A tuple containing:
                - The x-coordinates of the section centers.
                - The y-coordinates of the section centers.
        """
        angle_rad = np.pi / 2 - np.radians(strike)
        return (
            center_x + section_centers * np.cos(angle_rad), 
            center_y + section_centers * np.sin(angle_rad)
        )

    def _cross_section(self) -> pd.DataFrame:
        """
        Generate cross-sectional slices of seismic event data.

        Returns
        -------
        pd.DataFrame
            A concatenated dataframe containing seismic events for each 
            cross section. The dataframe includes:
            - All events within the defined thickness and depth range 
              for each section.
            - The ``'on_section_coords'`` column, representing the position 
              of events along the section.
            - The ``'section_id'`` column, indicating the cross-sectional 
              slice each event belongs to.
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

    @sync_signature('_plotter', 'plot_sections')
    def plot_sections(self, **kwargs):
        """
        Plots a cross-section of seismic events with customizable appearance.

        Parameters
        ----------
        color_by : str, optional
            Specifies the column in the DataFrame used to color the 
            seismic events. Default is ``None``, which applies a single color to 
            all points.

        cmap : str, optional
            The colormap to use for coloring events if ``color_by`` is specified. 
            Default is ``'jet'``.

        title : str, optional
            Title of the map. If ``None``, no title is displayed. Default is 
            ``Section``.

        hl_ms : int, optional
            If specified, highlights seismic events with a magnitude 
            greater than this value using different markers. Default is ``None``.

        hl_size : float, optional
            Size of the markers used for highlighted seismic events (if ``hl_ms`` 
            is specified). Default is 200.

        hl_marker : str, optional
            Marker style for highlighted events. Default is ``'*'``.

        hl_color : str, optional
            Color of the highlighted event markers. Default is ``'red'``.

        hl_edgecolor : str, optional
            Edge color for highlighted event markers. Default is ``'darkred'``.

        size : float or str, optional
            The size of the markers representing seismic events. If a string 
            is provided, it should refer to a column in the DataFrame to scale 
            point sizes proportionally. Default is 10.

        size_scale_factor : tuple[float, float], optional
            A tuple to scale marker sizes when ``size`` is based on a DataFrame 
            column. The first element scales the values, and the second element 
            raises them to a power. Default is ``(1, 3)``.            

        color : str, optional
            Default color for event markers when ``color_by`` is ``None``. 
            Default is ``'grey'``.

        edgecolor : str, optional
            Edge color for event markers. Default is ``'black'``.

        alpha : float, optional
            Transparency level for markers, ranging from 0 (transparent) to 1 
            (opaque). Default is 0.50.

        legend : str, optional
            Text for the legend describing the seismic events. If ``None``, 
            no legend is displayed. Default is ``None``.

        legend_loc : str, optional
            Location of the legend for the seismic event markers. 
            Default is ``'lower left'``.

        size_legend : bool, optional
            If ``True``, displays a legend that explains marker sizes. Default 
            is ``False``.
            
        size_legend_loc : str, optional
            Location of the size legend when ``size_legend`` is ``True``. Default 
            is ``'lower right'``.

        scale_legend: bool, optional
            If ``True``, displays a legend that shows a scale bar on the plot to 
            indicate real-world distances. Default is ``False``.

        scale_legend_loc : str, optional
              Location of the size legend when ``scale_legend`` is ``True``. 
              Default is ``'lower right'``.

        ylabel : str, optional
            Label for the y-axis. Default is ``'Depth [km]'``.

        xlim : tuple[float, float], optional
            Distance from center limits for the map's horizontal extent. If 
            ``None``, the limits are determined automatically based on the 
            ``map_length`` attribute. Default is ``None``.

        ylim : tuple[float, float], optional
            Depth limits for the map's vertical extent. If ``None``, 
            the limits are determined automatically based on the ``depth_range``
            attribute. Default is ``None``.

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(12, 6)``.

        facecolor : tuple[str, str], optional
            Tuple specifying the background colors of the plot, with two 
            values for gradient-like effects. Default is ``('#F0F0F0', '#FFFFFF')``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is 
            ``'cross_section'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). 
            Default is ``'jpg'``.

        Returns
        -------
        None
            A cross-section plot.
        
        Examples
        --------
        An example of a seismic map generated using this method:

        .. image:: https://imgur.com/a/bOEbe7Y.jpg
            :align: center

        For detailed examples and step-by-step instructions on how to plot this 
        cross section, refer to the tutorials page in the documentation.
        """
        self._plotter.plot_sections(**kwargs)

    @sync_signature('_plotter', 'plot_section_lines')
    def plot_section_lines(self, **kwargs):
        """
        Visualizes section lines on a map.

        Parameters
        ----------
        highligt_mag : float, optional
            If specified, highlights all seismic events (that are present 
            in your sections) with a magnitude greater than this value by 
            plotting them as stars.
            
        title : str, optional
            The title to be displayed above the map. If not provided, the 
            map will have no title.

        size : float, optional
            The size of the markers used to represent seismic events on 
            the map. Default is 0.5.

        color : str, optional
            The color used to fill the seismic event markers. Default is 
            'lightgrey'.

        edgecolor : str, optional
            The color used for the edges of the seismic event markers. 
            Default is 'grey'.

        alpha : float, optional
            The transparency level of the markers. A value between 0 and 
            1, where 1 is fully opaque and 0 is fully transparent. 
            Default is 0.5.

        legend : str, optional
            Text for the legend describing the plotted seismic events. If 
            None, no legend is displayed.

        xlim : Tuple[float, float], optional
            A tuple specifying the minimum and maximum longitude values 
            to set the map extent horizontally. If not provided, the 
            extent will be set automatically based on the data.

        ylim : Tuple[float, float], optional
            A tuple specifying the minimum and maximum latitude values 
            to set the map extent vertically. If not provided, the extent 
            will be set automatically based on the data.

        inset : bool, optional
            Determines whether to include an inset map showing a broader 
            geographic context. Defaults to True.

        inset_buffer : float, optional
            A factor that enlarges the buffer area around the selection 
            shape in the inset, enhancing visibility and context. The 
            default value is 3.

        bounds_res : str, optional
            The resolution for the geographical boundaries (such as 
            coastlines and borders) on the map. Common values are '10m', 
            '50m', or '110m', with '10m' being the most detailed and 
            '110m' the least.

        bmap_res : int, optional
            The zoom level or resolution for the underlying map image 
            (e.g., satellite or terrain map). A higher value provides a 
            more detailed map image, with typical values ranging from 1 
            (very coarse) to 12 (very detailed).

        save_figure : bool, optional
            If set to True, the function saves the generated plots using 
            the provided base name and file extension. The default is 
            False.

        save_name : str, optional
            The base name used for saving figures when `save_figure` is 
            True. It serves as the prefix for file names. The default 
            base name is 'section'.

        save_extension : str, optional
            The file extension to use when saving figures, such as 'jpg', 
            'png', etc... The default extension is 'jpg'.
        """
        self._plotter.plot_section_lines(**kwargs)