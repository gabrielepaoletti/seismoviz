import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from seismoviz.utils import convert_to_utm
from seismoviz.components.common import styling
from seismoviz.components.common.map_plotter import MapPlotter
from seismoviz.components.common.base_plotter import BasePlotter
from seismoviz.components.analysis.statistical import StatisticalAnalyzer


class CatalogPlotter:
    def __init__(
            self,
            catalog: type,
            projection = ccrs.Mercator(),
            transform = ccrs.PlateCarree()
    ) -> None:
        self.ct = catalog
        self.mp = MapPlotter(
            projection=projection,
            transform=transform
        )
        self.bp = BasePlotter()
        self.sa = StatisticalAnalyzer(catalog)

    @staticmethod
    def _format_x_axis_time(ax) -> None:
        """
        Format the x-axis based on the displayed range of the axis.

        Parameters
        ----------
        ax : matplotlib axis
            The axis to format.
        """
        x_min, x_max = mdates.num2date(ax.get_xlim())
        time_range = x_max - x_min
        
        if time_range.days > 3650:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))        
        elif time_range.days > 730:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        elif time_range.days > 60:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[15]))
        else:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.xaxis.set_minor_locator(mdates.DayLocator())

        plt.xticks(rotation=30, ha='right')

    def _get_distance_from_center(
            self,
            center: tuple[float, float],
            strike: int
    ) -> pd.DataFrame:
        """
        Calculates the distance of each seismic event from a specified center 
        point along a defined strike direction.

        Parameters
        ----------
        center : tuple[float, float]
            The (longitude, latitude) coordinates of the center point in degrees.

        strike : int
            The strike angle in degrees, measured clockwise from north. Defines 
            the direction along which distances are calculated.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the seismic events, with an additional column 
            `'distance'` representing the calculated distance of each event from 
            the center along the strike direction.
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

        df_copy = self.ct.data.copy()
        df_copy['distance'] = -distance_from_center

        return df_copy

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
            size: float = 10,
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
            projection = ccrs.Mercator(),
            transform = ccrs.PlateCarree(),
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

        if isinstance(size, (int, float)):
            plt_size = size
        elif isinstance(size, str):
            plt_size = (self.ct.data[size]*size_scale_factor[0]) ** size_scale_factor[1]
        else:
            raise ValueError(
                "The 'size' parameter must be a scalar or a column from your data."
            )

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
            large_quakes = self.ct.data[self.ct.data['mag'] > hl_ms]
            self.mp.scatter(
                x=large_quakes.lon, y=large_quakes.lat, c=hl_color, s=hl_size, 
                marker=hl_marker, edgecolor=hl_edgecolor, linewidth=0.75, 
                label=f'Events M > {hl_ms}'
            )

        if legend:
            leg = plt.legend(
                loc=legend_loc,
                fancybox=False,
                edgecolor='black'
            )
            leg.legend_handles[0].set_sizes([50])
            
            if hl_ms is not None:
                leg.legend_handles[1].set_sizes([90])
            
            self.mp.ax.add_artist(leg)
            
            if isinstance(size, str) and size_legend:
                min_size = np.floor(min(self.ct.data[size]))
                max_size = np.ceil(max(self.ct.data[size]))
                size_values = [min_size, (min_size + max_size)/2, max_size]
                size_legend_labels = [
                    f"{'M' if size == 'mag' else 'D' if size == 'depth' else size} {v}" 
                    for v in size_values
                ]

                
                size_handles = [
                    plt.scatter(
                        [], [], s=(v*size_scale_factor[0]) ** size_scale_factor[1], 
                        facecolor='white', edgecolor='black', alpha=alpha, label=label)
                    for v, label in zip(size_values, size_legend_labels)
                ]
                
                leg2 = plt.legend(
                    handles=size_handles,
                    loc=size_legend_loc,
                    fancybox=False,
                    edgecolor='black',
                    ncol=len(size_values)
                )
                
                self.mp.ax.add_artist(leg2)

        if inset:
            self.mp.inset(main_extent, buffer=inset_buffer, bounds_res=bounds_res)

        if save_figure:
            self.bp.save_figure(save_name, save_extension)

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
            The (longitude, latitude) coordinates of the center point for distance calculation.

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
            If ``True``, displays a legend that explains marker sizes. Default is ``False``.
            
        size_legend_loc : str, optional
            Location of the size legend when ``size_legend`` is ``True``. Default is 
            ``'lower right'``.

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
            Base name for the file if `save_figure` is ``True``. Default is ``'space_time'``.

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
        self.bp.set_style(styling.DEFAULT)

        df_with_dist = self._get_distance_from_center(center=center, strike=strike)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title('Space-time distribution', fontweight='bold')

        if isinstance(size, (int, float)):
            plt_size = size
        elif isinstance(size, str):
            plt_size = (df_with_dist[size] * size_scale_factor[0]) ** size_scale_factor[1]
        else:
            raise ValueError(
                "The 'size' parameter must be a scalar or a column from your data."
            )

        if color_by:
            fig.set_figwidth(fig_size[0] + 2)
            self.bp.plot_with_colorbar(
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
            large_quakes = df_with_dist[df_with_dist['mag'] > hl_ms]
            ax.scatter(
                x=large_quakes.time, y=large_quakes.distance, c=hl_color, s=hl_size, 
                marker=hl_marker, edgecolor=hl_edgecolor, linewidth=0.75, 
                label=f'Events M > {hl_ms}'
            )

        if legend:
            leg = plt.legend(loc=legend_loc, fancybox=False, edgecolor='black')
            leg.legend_handles[0].set_sizes([50])

            if hl_ms is not None:
                leg.legend_handles[1].set_sizes([90])

            ax.add_artist(leg)

            if isinstance(size, str) and size_legend:
                min_size = np.floor(min(df_with_dist[size]))
                max_size = np.ceil(max(df_with_dist[size]))
                size_values = [min_size, (min_size + max_size) / 2, max_size]
                size_legend_labels = [
                    f"{'M' if size == 'mag' else 'D' if size == 'depth' else size} {v}" 
                    for v in size_values
                ]

                size_handles = [
                    plt.scatter(
                        [], [], s=(v * size_scale_factor[0]) ** size_scale_factor[1], 
                        facecolor='white', edgecolor='black', alpha=alpha, label=label
                    )
                    for v, label in zip(size_values, size_legend_labels)
                ]

                leg2 = plt.legend(
                    handles=size_handles,
                    loc=size_legend_loc,
                    fancybox=False,
                    edgecolor='black',
                    ncol=len(size_values),
                    borderpad=1.2
                )

                ax.add_artist(leg2)

        if xlim is not None:
            ax.set_xlim(
                mdates.date2num(pd.to_datetime(xlim[0])),
                mdates.date2num(pd.to_datetime(xlim[1]))
            )
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_ylabel('Distance from center [km]')
        self._format_x_axis_time(ax)

        ax.grid(True, alpha=0.25, axis='y', linestyle=':')

        if save_figure:
            self.bp.save_figure(save_name, save_extension)

        plt.show()
        self.bp.reset_style()

    def plot_magnitude_time(
            self,
            color_by: str = None,
            cmap: str = 'jet',
            size: float | str = 10,
            size_scale_factor: tuple[float, float] = (1, 2),
            color: str = 'grey',
            edgecolor: str = 'black',
            alpha: float = 0.75,
            size_legend: bool = False,
            size_legend_loc: str = 'upper right',
            fig_size: tuple[float, float] = (10, 5),
            save_figure: bool = False,
            save_name: str = 'magnitude_time', 
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plots seismic event magnitudes over time.

        Parameters
        ----------            
        color_by : str, optional
            Specifies the column in the DataFrame used to color the 
            seismic events. Default is ``None``, which applies a single color to 
            all points.

        cmap : str, optional
            The colormap to use for coloring events if ``color_by`` is specified. 
            Default is ``'jet'``.

        size : float or str, optional
            The size of the markers representing seismic events. If a string 
            is provided, it should refer to a column in the DataFrame to scale 
            point sizes proportionally. Default is 10.

        size_scale_factor : tuple[float, float], optional
            A tuple to scale marker sizes when ``size`` is based on a DataFrame 
            column. The first element scales the values, and the second element 
            raises them to a power. Default is (1, 2).

        color : str, optional
            Default color for event markers when ``color_by`` is ``None``. 
            Default is ``'grey'``.

        edgecolor : str, optional
            Edge color for event markers. Default is ``'black'``.

        alpha : float, optional
            Transparency level for markers, ranging from 0 (transparent) to 1 
            (opaque). Default is 0.75.

        size_legend : bool, optional
            If ``True``, displays a legend that explains marker sizes. Default is 
            ``False``.
            
        size_legend_loc : str, optional
            Location of the size legend when ``size_legend`` is ``True``. Default 
            is ``'upper right'``. 

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(10, 5)``.
            
        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is 
            ``'magnitude_time'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        None
            A magnitude-time plot.

        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Plot magnitude-time distribution
            catalog.plot_magnitude_time(
                color_by='depth',
                size='depth',
                cmap='YlOrRd',
            )
        
        .. image:: https://imgur.com/qYguHD1.jpg
            :align: center
        """
        self.bp.set_style(styling.DEFAULT)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title('Magnitude-time distribution', fontweight='bold')
        
        if isinstance(size, (int, float)):
            plt_size = size
        elif isinstance(size, str):
            plt_size = (self.ct.data[size]*size_scale_factor[0]) ** size_scale_factor[1]
        else:
            raise ValueError(
                "The 'size' parameter must be a scalar or a column from your data."
            )

        if color_by:
            fig.set_figwidth(fig_size[0] + 2)
            self.bp.plot_with_colorbar(
                ax=ax,
                data=self.ct.data,
                x='time',
                y='mag',
                color_by=color_by,
                cmap=cmap,
                size=plt_size,
                edgecolor=edgecolor,
                alpha=alpha,
                cbar_orientation='vertical',
                cbar_shrink=1,
                cbar_aspect=30,
                cbar_pad=0.03
            )
        else:
            ax.scatter(
                x=self.ct.data.time,
                y=self.ct.data.mag,
                c=color,
                s=plt_size,
                edgecolor=edgecolor,
                alpha=alpha,
                lw=0.25
            )

        if size_legend:   
            if isinstance(size, str):
                min_size = np.floor(min(self.ct.data[size]))
                max_size = np.ceil(max(self.ct.data[size]))
                size_values = [min_size, (min_size + max_size)/2, max_size]
                size_legend_labels = [
                    f"{'M' if size == 'mag' else 'D' if size == 'depth' else size} {v}" 
                    for v in size_values
                ]

                size_handles = [
                    plt.scatter(
                        [], [], s=(v*size_scale_factor[0]) ** size_scale_factor[1], 
                        facecolor='white', edgecolor='black', alpha=alpha, label=label
                    )
                    for v, label in zip(size_values, size_legend_labels)
                ]
                
                plt.legend(
                    handles=size_handles,
                    loc=size_legend_loc,
                    fancybox=False,
                    edgecolor='black',
                    ncol=len(size_values),
                )

        self._format_x_axis_time(ax)
        ax.grid(True, alpha=0.25, axis='y', linestyle=':')

        if save_figure:
            self.bp.save_figure(save_name, save_extension)
        
        plt.show()
        self.bp.reset_style()

    def plot_event_timeline(
            self,
            fig_size: tuple[float, float] = (10, 5), 
            save_figure: bool = False,
            save_name: str = 'event_timeline', 
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plots a timeline of seismic events to visualize the cumulative 
        number of events over time.  

        Parameters
        ----------  
        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(10, 5)``.
            
        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is 
            ``'event_timeline'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        None
            A event timeline plot.

        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Plot the event timeline
            catalog.plot_event_timeline()
        
        .. image:: https://imgur.com/FNnTzAV.jpg
            :align: center
        """
        self.bp.set_style(styling.DEFAULT)
        
        events_sorted = self.ct.data.sort_values('time')
        time_data = pd.to_datetime(events_sorted['time'])

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title('Event timeline', fontweight='bold')
        ax.plot(time_data, np.arange(len(events_sorted)), color='black')

        self._format_x_axis_time(ax)

        ax.set_ylabel('Cumulative no. of events')
        ax.set_ylim(0)
        ax.grid(True, axis='x', linestyle=':', linewidth=0.25)
        
        plt.tight_layout()
        if save_figure:
            self.bp.save_figure(save_name, save_extension)
        
        plt.show()
        self.bp.reset_style()

    def plot_attribute_distributions(
            self,
            fig_size: tuple[float, float] = (10, 6),
            save_figure: bool = False,
            save_name: str = 'attribute_distributions', 
            save_extension: str = 'jpg'
    ) -> None:
        """
        Visualizes the distribution of key attributes in the seismic event 
        catalog.
        
        Parameters
        ----------
        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is 
            ``'attribute_distributions'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        None
            A plot showing the distribution of the main attributes of the 
            catalog.

        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Plot attribute distributions
            catalog.plot_attribute_distributions()
        
        .. image:: https://imgur.com/UfucKab.jpg
            :align: center
        """
        self.bp.set_style(styling.DEFAULT)

        rows, cols = 2, 2
        fig, ax = plt.subplots(rows, cols, figsize=fig_size, sharey=True)
        plt.suptitle('Attribute distributions', fontsize=18, fontweight='bold')
        labels = {'lon': 'Longitude', 'lat': 'Latitude', 'mag': 'Magnitude',
                  'depth': 'Depth'}

        for i, (attribute, label) in enumerate(labels.items()):
            row, col = divmod(i, cols)
            ax[row, col].hist(
                self.ct.data[attribute], bins=50, color='silver', 
                edgecolor='black', linewidth=0.5
            )
            ax[row, col].set_xlabel(label)
            ax[row, col].set_ylabel('Number of events' if col == 0 else '')
            ax[row, col].grid(True, alpha=0.25, axis='y', linestyle=':')

        plt.tight_layout()
        if save_figure:
            self.bp.save_figure(save_name, save_extension)
        
        plt.show()
        self.bp.reset_style()

    def plot_interevent_time(
            self,
            plot_vs: str = 'time',
            plot_event_timeline: bool = True,
            et_axis_color: str = 'red',
            et_line_color: str = 'red',
            color_by: str = None,
            cmap: str = 'jet',
            size: float | str = 10,
            size_scale_factor: tuple[float, float] = (1, 2),
            yscale: str = 'log',
            color: str = 'grey',
            edgecolor: str = 'black',
            alpha: float = 0.75,
            size_legend: bool = False,
            size_legend_loc: str = 'upper right',
            fig_size: tuple[float, float] = (10, 5),
            save_figure: bool = False,
            save_name: str = 'interevent_time',
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plots inter-event times against any specified attribute.

        .. note::
            After executing this method, a new column named ``interevent_time`` 
            will be created in the ``catalog.data`` DataFrame. This column will 
            contain the calculated inter-event times, making the data accessible 
            for further analysis or visualization.

        Parameters
        ----------
        plot_vs : str, optional
            Specifies the column to plot inter-event times against. Default is 
            ``'time'``.        

        plot_event_timeline : bool, optional
            If ``True`` and ``plot_vs='time'``, adds a secondary y-axis (``twiny``) 
            to plot a cumulative event timeline. The timeline represents the cumulative 
            number of events over time. Default is ``True``.

        et_axis_color : str, optional
            Specifies the color of the secondary y-axis (event timeline axis), 
            including the ticks, labels, and axis line. Default is ``'red'``.

        et_line_color : str, optional
            Specifies the color of the line representing the cumulative number 
            of events on the secondary y-axis. Default is ``'red'``.
            
        color_by : str, optional
            Specifies the column in the DataFrame used to color the 
            seismic events. Default is ``None``, which applies a single color to 
            all points.

        cmap : str, optional
            The colormap to use for coloring events if ``color_by`` is specified. 
            Default is ``'jet'``.

        size : float or str, optional
            The size of the markers representing seismic events. If a string 
            is provided, it should refer to a column in the DataFrame to scale 
            point sizes proportionally. Default is 10.

        size_scale_factor : tuple[float, float], optional
            A tuple to scale marker sizes when ``size`` is based on a DataFrame 
            column. The first element scales the values, and the second element 
            raises them to a power. Default is (1, 2).

        yscale : str, optional
            Specifies the scale of the y-axis. Common options include ``'linear'``, 
            ``'log'``, ``'symlog'``, and ``'logit'``. Default is ``'linear'``.

        color : str, optional
            Default color for event markers when ``color_by`` is ``None``. 
            Default is ``'grey'``.

        edgecolor : str, optional
            Edge color for event markers. Default is ``'black'``.

        alpha : float, optional
            Transparency level for markers, ranging from 0 (transparent) to 1 
            (opaque). Default is 0.75.

        size_legend : bool, optional
            If ``True``, displays a legend that explains marker sizes. Default is 
            ``False``.
            
        size_legend_loc : str, optional
            Location of the size legend when ``size_legend`` is ``True``.
            Default is ``'upper right'``. 

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(10, 5)``.
            
        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is 
            ``'interevent_time'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        None
            A plot of inter-event times.

        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Plot inter-event time distribution
            catalog.plot_interevent_time(
                plot_vs='time',
                plot_event_timeline=True,
                size=5
            )
        
        .. image:: https://imgur.com/Nx79ICZ.jpg
            :align: center
        """
        self.sa.calculate_interevent_time(unit='sec')

        self.bp.set_style(styling.DEFAULT)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title('Inter-event time distribution')

        if isinstance(size, (int, float)):
            plt_size = size
        elif isinstance(size, str):
            plt_size = (self.ct.data[size] * size_scale_factor[0]) ** size_scale_factor[1]
        else:
            raise ValueError(
                "The 'size' parameter must be a scalar or a column from your data."
            )

        if color_by:
            fig.set_figwidth(fig_size[0] + 2)
            self.bp.plot_with_colorbar(
                ax=ax,
                data=self.ct.data,
                x=plot_vs,
                y='interevent_time',
                color_by=color_by,
                cmap=cmap,
                size=plt_size,
                edgecolor=edgecolor,
                alpha=alpha,
                cbar_orientation='vertical',
                cbar_shrink=1,
                cbar_aspect=30,
                cbar_pad=0.115 if plot_event_timeline else 0.03
            )
        else:
            ax.scatter(
                x=self.ct.data[plot_vs],
                y=self.ct.data['interevent_time'],
                c=color,
                s=plt_size,
                edgecolor=edgecolor,
                alpha=alpha,
                lw=0.25
            )

        if size_legend:
            if isinstance(size, str):
                min_size = np.floor(min(self.ct.data[size]))
                max_size = np.ceil(max(self.ct.data[size]))
                size_values = [min_size, (min_size + max_size) / 2, max_size]
                size_legend_labels = [
                    f"{'M' if size == 'mag' else 'D' if size == 'depth' else size} {v}" 
                    for v in size_values
                ]


                size_handles = [
                    plt.scatter(
                        [], [], s=(v * size_scale_factor[0]) ** size_scale_factor[1],
                        facecolor='white', edgecolor='black', alpha=alpha, label=label
                    )
                    for v, label in zip(size_values, size_legend_labels)
                ]

                plt.legend(
                    handles=size_handles,
                    loc=size_legend_loc,
                    fancybox=False,
                    edgecolor='black',
                    ncol=len(size_values),
                )

        if plot_vs == 'time':
            self._format_x_axis_time(ax)

            if plot_event_timeline:
                self.bp.set_style(styling.CROSS_SECTION)
                
                ax_twin = ax.twinx()
                twin_axis_color = f'tab:{et_axis_color}'
                ax_twin.spines['right'].set_color(twin_axis_color)
                ax_twin.set_ylabel(
                    'Cumulative no. of events', color=twin_axis_color
                )
                ax_twin.tick_params(
                    axis='y', color=twin_axis_color, labelcolor=twin_axis_color
                )

                timeline = self.ct.data['time']
                cumulative_events = range(1, len(timeline) + 1)
                ax_twin.plot(timeline, cumulative_events, color=et_line_color)

        ax.grid(True, alpha=0.25, axis='y', linestyle=':')
    
        xlabel_map = {
            'time': None,
            'mag': 'Magnitude',
            'lat': 'Latitude [°]',
            'lon': 'Longitude [°]',
            'depth': 'Depth [Km]'
        }

        ax.set_yscale(yscale)
        ax.set_ylabel('Interevent Time [s]')
        ax.set_xlabel(xlabel_map[plot_vs])

        if save_figure:
            self.bp.save_figure(save_name, save_extension)

        plt.show()
        self.bp.reset_style()


class SubCatalogPlotter:
    def __init__(self, sub_catalog: type) -> None:
            self.sc = sub_catalog
            self.mp = MapPlotter()
            self.bp = BasePlotter()
    
    def plot_on_section(
            self, 
            title: str = None,
            color_by: str = None,
            cmap: str = 'jet',
            hl_ms: float = 300,
            hl_size: float = 200,
            hl_marker: str = '*',
            hl_color: str = 'red',
            hl_edgecolor: str = 'darkred',
            size: float = 1,
            size_scale_factor: tuple[float, float] = (1, 2),
            color: str = 'grey',
            edgecolor: str = 'black',
            alpha: float = 0.75,
            legend: str = None,
            legend_loc: str = 'lower left',
            size_legend: bool = False,
            size_legend_loc: str = 'upper right',
            scale_legend: bool = True,
            scale_legend_loc: str  = 'lower right',
            ylabel: str = 'Depth [km]',
            normalize: bool = True
        ) -> None:
        if self.sc.selected_from != 'CrossSection':
            raise ValueError('To be plotted on-section, the SubCatalog must be '
                             'sampled from a CrossSection object.')
        self.bp.set_style(styling.DEFAULT)
        
        osc = self.sc.data.on_section_coords
        if normalize:
            osc = osc - osc.median()
        
        if isinstance(size, (int, float)):
            plt_size = size
        elif isinstance(size, str):
            plt_size = (self.sc.data[size] * size_scale_factor[0]) ** size_scale_factor[1]
        else:
            raise ValueError(
                "The 'size' parameter must be a scalar or a column from your data."
            )

        fig, ax = plt.subplots(figsize=(12, 6))

        if color_by:
            fig.set_figheight(8)
            self.bp.plot_with_colorbar(
                ax=ax,
                data=self.sc.data,
                x='on_section_coords',
                y='depth',
                color_by=color_by,
                cmap=cmap,
                edgecolor=edgecolor,
                size=plt_size,
                alpha=alpha,
                legend=legend,
                cbar_pad=0.05,
            )
        else:
            ax.scatter(
                self.sc.data.on_section_coords,
                self.sc.data.depth,
                color=color, 
                edgecolor=edgecolor,
                s=plt_size, 
                alpha=alpha,
                linewidth=0.25,
                label=legend
            )
    
        if title:
            ax.set_title(f'{title}', fontweight='bold')
        
        if hl_ms is not None:
            large_quakes = self.sc.data[self.sc.data['mag'] > hl_ms]
            ax.scatter(
                x=large_quakes.on_section_coords, y=large_quakes.depth, c=hl_color, 
                s=hl_size, marker=hl_marker, edgecolor=hl_edgecolor, linewidth=0.75,
                label=f'Events M > {hl_ms}'
            )
        
        ax.set_ylabel(ylabel)
        #ax.xaxis.set_visible(False)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, linestyle=':')

        if scale_legend:
            scale_length = ((osc.max() - osc.min()) / 2) / 5
            scale_label = f'{scale_length:.1f} km'

            scalebar = AnchoredSizeBar(
                transform=ax.transData,
                size=scale_length,
                label=scale_label,
                loc=scale_legend_loc,
                sep=5,
                color='black',
                frameon=False,
                size_vertical=(self.sc.data.depth.max() - self.sc.data.depth.min()) / 100,
                fontproperties=fm.FontProperties(size=10, weight='bold')
            )

            ax.add_artist(scalebar)

        if legend:
            leg = plt.legend(loc=legend_loc, fancybox=False, edgecolor='black')
            leg.legend_handles[0].set_sizes([50])

            if hl_ms is not None:
                leg.legend_handles[1].set_sizes([90])

            ax.add_artist(leg)

            if isinstance(size, str) and size_legend:
                min_size = np.floor(min(self.sc.data[size]))
                max_size = np.ceil(max(self.sc.data[size]))
                size_values = [min_size, (min_size + max_size) / 2, max_size]
                size_legend_labels = [
                    f"{'M' if size == 'mag' else 'D' if size == 'depth' else size} {v}" 
                    for v in size_values
                ]


                size_handles = [
                    plt.scatter(
                        [], [], s=(v * size_scale_factor[0]) ** size_scale_factor[1],
                        facecolor='white', edgecolor='black', alpha=alpha, label=label
                    )
                    for v, label in zip(size_values, size_legend_labels)
                ]

                leg2 = plt.legend(
                    handles=size_handles,
                    loc=size_legend_loc,
                    fancybox=False,
                    edgecolor='black',
                    ncol=len(size_values),
                )

                ax.add_artist(leg2)
        
        plt.show()
        self.bp.reset_style()