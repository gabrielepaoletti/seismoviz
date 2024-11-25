import numpy as np
import pandas as pd

from seismoviz.utils import convert_to_utm
from seismoviz.components.catalog import Catalog
from seismoviz.internal.decorators import sync_signature
from seismoviz.internal.mixins import DunderMethodMixin, GeospatialMixin
from seismoviz.components.plotters.crs_plotter import CrossSectionPlotter

from numpy.typing import ArrayLike


class CrossSection(GeospatialMixin, DunderMethodMixin):
    """
    Represents a seismic cross section.

    Attributes
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

    _plotter : CrossSectionPlotter
        An internal object for generating plots of the cross-section and related visuals.

    data : pd.DataFrame
        A DataFrame containing the seismic event data for each cross-sectional slice.

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
        .. code-block:: python
        
            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Create cross section object
            cs = sv.create_cross_section(
                catalog=catalog,        
                center=(13.12, 42.83),  
                num_sections=(0,0),     
                thickness=2,            
                strike=155,             
                map_length=40,          
                depth_range=(0, 10)     
            )

            # Visualize the cross-section
            cs.plot_sections(
                color_by='time',        
                cmap='Blues',           
                size='mag',             
                edgecolor='black',
                hl_ms=5,
                hl_size=300,
                legend='Seismicity',
                legend_loc='upper left',
                scale_legend_loc='upper right'  
            )  

        .. image:: https://imgur.com/juiawDc.jpg
            :align: center
        """
        self._plotter.plot_sections(**kwargs)

    @sync_signature('_plotter', 'plot_section_lines')
    def plot_section_lines(self, **kwargs):
        """
        Visualizes section lines on a map.

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
            Title of the map. If ``None``, no title is displayed. Default is ``None``.

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
            (opaque). Default is 0.75.

        sl_color : str, optional
            Line color for section lines. Default is ``'blue'``.

        sl_linewidth : float, optional
            Line width for section lines. Default is 1.5.

        sl_text_size : float, optional
            Font size for section line labels. Default is 10.

        sl_text_color : str, optional
            Text color for section line labels. Default is ``'white'``.

        sl_text_weight : str, optional
            Text weight for section line labels. Default is ``'bold'``.

        sl_box_style : str, optional
            Style for the bounding box around section line labels. Default is
            ``'circle'``.

        sl_box_color : str, optional
            Fill color for the bounding box around section line labels. Default 
            is ``'black'``.

        sl_box_edgecolor : str, optional
            Edge color for the bounding box around section line labels. Default 
            is ``'black'``.

        sl_box_pad : float, optional
            Padding inside the bounding box for section line labels. Default is 
            0.3.

        legend : str, optional
            Text for the legend describing the seismic events. If ``None``, 
            no legend is displayed. Default is ``None``.

        legend_loc : str, optional
            Location of the legend for the seismic event markers. 
            Default is ``'lower left'``.

        size_legend : bool, optional
            If ``True``, displays a legend that explains marker sizes. Default is 
            ``False``.
            
        size_legend_loc : str, optional
            Location of the size legend when ``size_legend`` is ``True``. Default 
            is ``'lower right'``.            

        xlim : tuple[float, float], optional
            Longitude limits for the map's horizontal extent. If ``None``, 
            the limits are determined automatically based on the data. 
            Default is ``None``.

        ylim : tuple[float, float], optional
            Latitude limits for the map's vertical extent. If ``None``, 
            the limits are determined automatically based on the data. 
            Default is ``None``.

        terrain_cmap : str, optional
            The colormap to be applied to the terrain layer. Defaults to ``'gray_r'``.            

        terrain_style : str, optional
            The style of the terrain background for the map. Common values 
            include ``'satellite'``, ``'terrain'`` or ``'street'``.Defaults to 
            ``'satellite'``.

        terrain_alpha : float, optional
            The transparency level for the terrain layer, where 0 is fully 
            transparent and 1 is fully opaque. Defaults to 0.35.     

        projection : cartopy.crs projection, optional
            The map projection used to display the map. Defaults to 
            ``ccrs.Mercator()``.

        transform : cartopy.crs projection, optional
            The coordinate reference system of the data to be plotted. 
            Defaults to ``ccrs.PlateCarree()``.

        inset : bool, optional
            If ``True``, adds an inset map for broader geographic context. 
            Default is ``False``.

        inset_buffer : float, optional
            Scaling factor for the area surrounding the selection shape 
            in the inset map. Default is 3.

        bounds_res : str, optional
            Resolution of geographical boundaries (coastlines, borders) 
            on the map. Options are ``'10m'`` (highest resolution), ``'50m'``, 
            and ``'110m'`` (lowest resolution). Default is '50m'.

        bmap_res : int, optional
            Resolution level for the base map image (e.g., satellite or 
            terrain). Higher values provide more detail. Default is 12.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is ``'map'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        None
            A map showing section lines.
        
        Examples
        --------
        .. code-block:: python
        
            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Create cross section object
            cs = sv.create_cross_section(
                catalog=catalog,        
                center=(13.12, 42.83),  
                num_sections=(2, 2),     
                thickness=2,            
                strike=155,             
                map_length=40,          
                depth_range=(0, 10),
                section_distance=8    
            )

            # Plot section traces on a map
            cs.plot_section_lines(
                title='Section lines',
                size='mag',
                color='lightgrey',
                hl_ms=5,
                hl_size=300,
                sl_box_style='square',
                sl_box_color='white',
                sl_text_color='black',
                legend='Tan et al. 2021',
                inset=True,
                bmap_res=12
            )

        .. image:: https://imgur.com/7n4ZAFL.jpg
            :align: center
        """
        self._plotter.plot_section_lines(**kwargs)