import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import seismoviz.components.visualization.utils.plot_utils as pu

from seismoviz.utils import convert_to_utm
from seismoviz.components.visualization.utils import styling
from seismoviz.components.visualization.plot_common import CommonPlotter
from seismoviz.components.visualization.utils.map_plotter import MapPlotter


class CatalogPlotter(CommonPlotter):
    """
    Provides plotting methods for Catalog objects.
    """

    def __init__(
        self,
        catalog: type,
        projection=ccrs.Mercator(),
        transform=ccrs.PlateCarree()
    ) -> None:
        super().__init__(catalog)
        self.ct = catalog
        self.mp = MapPlotter(
            projection=projection,
            transform=transform
        )

    def plot_map(
        self,
        color_by: str = None,
        cmap: str = 'jet',
        title: str = None,
        hl_ms: int = None,
        hl_size: float = 200,
        hl_marker: str = '*',
        hl_color: str = 'red',
        hl_edgecolor: str = 'darkred',
        size: float | str = 10,
        size_scale_factor: tuple[int, int] = (1, 3),
        color: str = 'grey',
        edgecolor: str = 'black',
        alpha: float = 0.75,
        legend: str = None,
        legend_loc: str = 'lower left',
        size_legend: bool = False,
        size_legend_loc: str = 'lower right',
        terrain_style: str = 'satellite',
        terrain_cmap: str = 'gray_r',
        terrain_alpha: str = 0.35,
        inset: bool = False,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        inset_buffer: float = 3,
        bounds_res: str = '50m',
        bmap_res: int = 5,
        projection=ccrs.Mercator(),
        transform=ccrs.PlateCarree(),
        save_figure: bool = False,
        save_name: str = 'map',
        save_extension: str = 'jpg'
    ) -> None:
        """
        Visualizes seismic events on a geographical map.

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
            ``None``.

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
            Location of the size legend when ``size_legend`` is ``True``. Default is 
            ``'lower right'``.

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
            A map with seismic events.

        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='global_seismic_catalog.csv')

            # Create a map showing earthquake locations
            catalog.plot_map(
                title='Global seismicity (M > 4.0)',
                color_by='depth',
                cmap='YlOrRd',
                size='mag',
                projection=ccrs.Robinson()
            )

        .. image:: https://imgur.com/0d6OA1L.jpg
            :align: center
        """
        self.mp.transform, self.mp.projection = transform, projection

        self.mp.fig, self.mp.ax = self.mp.create_base_map(
            terrain_style=terrain_style,
            terrain_cmap=terrain_cmap,
            terrain_alpha=terrain_alpha,
            bounds_res=bounds_res,
            bmap_res=bmap_res
        )
        main_extent = self.mp.extent(self.ct.data, xlim=xlim, ylim=ylim)

        plt_size = pu.process_size_parameter(size, self.ct.data, size_scale_factor)

        if color_by:
            self.mp.fig.set_figheight(10)
            self.mp.plot_with_colorbar(
                data=self.ct.data,
                x='lon',
                y='lat',
                color_by=color_by,
                cmap=cmap,
                edgecolor=edgecolor,
                size=plt_size,
                alpha=alpha,
                legend=legend
            )
        else:
            self.mp.scatter(
                x=self.ct.data.lon, y=self.ct.data.lat, c=color, s=plt_size,
                edgecolor=edgecolor, linewidth=0.25, alpha=alpha, label=legend
            )

        if title:
            self.mp.ax.set_title(title, fontweight='bold')

        if hl_ms is not None:
            pu.plot_highlighted_events(
                ax=self.mp.ax,
                data=self.ct.data,
                hl_ms=hl_ms,
                hl_size=hl_size,
                hl_marker=hl_marker,
                hl_color=hl_color,
                hl_edgecolor=hl_edgecolor,
                x='lon',
                y='lat'
            )

        if legend:
            leg = plt.legend(
                loc=legend_loc,
                fancybox=False,
                edgecolor='black'
            )
            leg.legend_handles[0].set_sizes([50])

            if hl_ms is not None:
                leg.legend_handles[-1].set_sizes([90])

            self.mp.ax.add_artist(leg)

            if isinstance(size, str) and size_legend:
                pu.create_size_legend(
                    ax=self.mp.ax,
                    size=size,
                    data=self.ct.data,
                    size_scale_factor=size_scale_factor,
                    alpha=alpha,
                    size_legend_loc=size_legend_loc
                )

        if inset:
            self.mp.inset(
                main_extent, buffer=inset_buffer, bounds_res=bounds_res
            )

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()

    def plot_space_time(
        self,
        center: tuple[float, float],
        strike: int,
        color_by: str = None,
        cmap: str = 'jet',
        hl_ms: int = None,
        hl_size: float = 200,
        hl_marker: str = '*',
        hl_color: str = 'red',
        hl_edgecolor: str = 'darkred',
        size: float | str = 10,
        size_scale_factor: tuple[float, float] = (1, 2),
        color: str = 'grey',
        edgecolor: str = 'black',
        alpha: float = 0.75,
        xlim: tuple[str, str] = None,
        ylim: tuple[float, float] = None,
        legend: str = None,
        legend_loc: str = 'lower right',
        size_legend: bool = False,
        size_legend_loc: str = 'upper right',
        fig_size: tuple[float, float] = (10, 5),
        save_figure: bool = False,
        save_name: str = 'space_time',
        save_extension: str = 'jpg'
    ) -> None:
        """
        Plots the space-time distribution of seismic events along a specified 
        strike direction.

        Parameters
        ----------
        center : tuple[float, float]
            The (longitude, latitude) coordinates of the center point for distance 
            calculation.

        strike : int
            The strike angle in degrees, measured clockwise from north. Defines 
            the direction along which distances are calculated.

        color_by : str, optional
            Specifies the column in the DataFrame used to color the 
            seismic events. Default is ``None``, which applies a single color to 
            all points.

        cmap : str, optional
            The colormap to use for coloring events if ``color_by`` is specified. 
            Default is ``'jet'``.

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
            raises them to a power. Default is ``(1, 2)``.

        color : str, optional
            Default color for event markers when ``color_by`` is ``None``. 
            Default is ``'grey'``.

        edgecolor : str, optional
            Edge color for event markers. Default is ``'black'``.

        alpha : float, optional
            Transparency level for markers, ranging from 0 (transparent) to 1 
            (opaque). Default is 0.75.

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

        xlim : tuple[float, float], optional
            Time limits for the x-axis as start and end date strings. Default 
            is ``None``.

        ylim : tuple[float, float], optional
            Limits for the y-axis (distance from center). Default is ``None``.

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(10, 5)``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is 
            ``'space_time'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        None
            A space-time plot.

        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Plot space-time distribution
            catalog.plot_space_time(
                center=(13.12, 42.83),
                strike=155,
                hl_ms=5,
                size=0.5,
                color='black',
                alpha=0.25,
            )

        .. image:: https://imgur.com/AgrhmOt.jpg
            :align: center  
        """
        pu.set_style(styling.DEFAULT)

        df_with_dist = self._get_distance_from_center(center=center, strike=strike)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title('Space-time distribution', fontweight='bold')

        plt_size = pu.process_size_parameter(size, df_with_dist, size_scale_factor)

        if color_by:
            fig.set_figwidth(fig_size[0] + 2)
            pu.plot_with_colorbar(
                ax=ax,
                data=df_with_dist,
                x='time',
                y='distance',
                color_by=color_by,
                cmap=cmap,
                size=plt_size,
                edgecolor=edgecolor,
                alpha=alpha,
                legend=legend,
                cbar_orientation='vertical',
                cbar_shrink=1,
                cbar_aspect=30,
                cbar_pad=0.03
            )
        else:
            ax.scatter(
                x=df_with_dist.time,
                y=df_with_dist.distance,
                c=color,
                s=plt_size,
                edgecolor=edgecolor,
                alpha=alpha,
                lw=0.25,
                label=legend
            )

        if hl_ms is not None:
            pu.plot_highlighted_events(
                ax=ax,
                data=df_with_dist,
                hl_ms=hl_ms,
                hl_size=hl_size,
                hl_marker=hl_marker,
                hl_color=hl_color,
                hl_edgecolor=hl_edgecolor,
                x='time',
                y='distance'
            )

        if legend:
            leg = plt.legend(loc=legend_loc, fancybox=False, edgecolor='black')
            leg.legend_handles[0].set_sizes([50])

            if hl_ms is not None:
                leg.legend_handles[-1].set_sizes([90])

            ax.add_artist(leg)

            if isinstance(size, str) and size_legend:
                pu.create_size_legend(
                    ax=ax,
                    size=size,
                    data=df_with_dist,
                    size_scale_factor=size_scale_factor,
                    alpha=alpha,
                    size_legend_loc=size_legend_loc
                )

        if xlim is not None:
            ax.set_xlim(
                pd.to_datetime(xlim[0]),
                pd.to_datetime(xlim[1])
            )
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_ylabel('Distance from center [km]')
        pu.format_x_axis_time(ax)

        ax.grid(True, alpha=0.25, axis='y', linestyle=':')

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()

    def _get_distance_from_center(
        self,
        center: tuple[float, float],
        strike: int
    ) -> pd.DataFrame:
        """
        Calculates the distance of each seismic event from a specified center
        point along a defined strike direction.
        """
        self._utmx, self._utmy = convert_to_utm(
            self.ct.data.lon, self.ct.data.lat, zone=self.ct.zone, units='km',
            ellps='WGS84', datum='WGS84'
        )
        center_utmx, center_utmy = convert_to_utm(
            center[0], center[1],
            zone=self.ct.zone, units='km', ellps='WGS84', datum='WGS84'
        )

        normal_ref = [
            np.cos(np.radians(strike)),
            -np.sin(np.radians(strike)), 0
        ]

        distance_from_center = (
            (self._utmy - center_utmy) * normal_ref[0] -
            (self._utmx - center_utmx) * normal_ref[1]
        )

        df_distance = self.ct.data.copy()
        df_distance['distance'] = -distance_from_center

        return df_distance