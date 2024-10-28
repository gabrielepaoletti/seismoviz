import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from seismoviz.utils import convert_to_utm
from seismoviz.plotters.common import styling
from seismoviz.plotters.common.map_plotter import MapPlotter
from seismoviz.plotters.common.base_plotter import BasePlotter


class CatalogPlotter:
    def __init__(self, catalog: type) -> None:
        self.ct = catalog
        self.mp = MapPlotter()
        self.bp = BasePlotter()

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
            size_legend: bool = True,
            size_legend_loc: str = 'lower right',
            inset: bool = True, 
            xlim: tuple[float, float] = None,
            ylim: tuple[float, float] = None, 
            inset_buffer: float = 3, 
            bounds_res: str = '50m', 
            bmap_res: int = 12, 
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
            seismic events (e.g., 'magnitude', 'time', or 'depth'). 
            Default is None, which applies a single color to all points.

        cmap : str, optional
            The colormap to use for coloring events if `color_by` is specified. 
            Default is 'jet'.

        title : str, optional
            Title of the map. If None, no title is displayed. Default is None.

        hl_ms : int, optional
            If specified, highlights seismic events with a magnitude 
            greater than this value using different markers. Default is None.

        hl_size : float, optional
            Size of the markers used for highlighted seismic events (if `hl_ms` 
            is specified). Default is 200.

        hl_marker : str, optional
            Marker style for highlighted events. Default is '*'.

        hl_color : str, optional
            Color of the highlighted event markers. Default is 'red'.

        hl_edgecolor : str, optional
            Edge color for highlighted event markers. Default is 'darkred'.

        size : float or str, optional
            The size of the markers representing seismic events. If a string 
            is provided, it should refer to a column in the DataFrame (e.g., 
            'mag') to scale point sizes proportionally. Default is 10.

        size_scale_factor : tuple[float, float], optional
            A tuple to scale marker sizes when `size` is based on a DataFrame 
            column. The first element scales the values, and the second element 
            raises them to a power. Default is (1, 3).

        color : str, optional
            Default color for event markers when `color_by` is None. 
            Default is 'grey'.

        edgecolor : str, optional
            Edge color for event markers. Default is 'black'.

        alpha : float, optional
            Transparency level for markers, ranging from 0 (transparent) to 1 
            (opaque). Default is 0.75.

        legend : str, optional
            Text for the legend describing the seismic events. If None, 
            no legend is displayed. Default is None.

        legend_loc : str, optional
            Location of the legend for the seismic event markers. 
            Default is 'lower left'.

        size_legend : bool, optional
            If True, displays a legend that explains marker sizes. Default is True.
            
        size_legend_loc : str, optional
            Location of the size legend when `size_legend` is True. 
            Default is 'lower right'.

        xlim : tuple[float, float], optional
            Longitude limits for the map's horizontal extent. If None, 
            the limits are determined automatically based on the data. 
            Default is None.

        ylim : tuple[float, float], optional
            Latitude limits for the map's vertical extent. If None, 
            the limits are determined automatically based on the data. 
            Default is None.

        inset : bool, optional
            If True, adds an inset map for broader geographic context. 
            Default is True.

        inset_buffer : float, optional
            Scaling factor for the area surrounding the selection shape 
            in the inset map. Default is 3.

        bounds_res : str, optional
            Resolution of geographical boundaries (coastlines, borders) 
            on the map. Options are '10m', '50m', and '110m', where '10m' 
            is the highest resolution and '110m' the lowest. Default is '50m'.

        bmap_res : int, optional
            Resolution level for the base map image (e.g., satellite or 
            terrain). Higher values provide more detail. Default is 12.

        save_figure : bool, optional
            If True, saves the plot to a file. Default is False.

        save_name : str, optional
            Base name for the file if `save_figure` is True. Default is 'map'.

        save_extension : str, optional
            File format for the saved figure (e.g., 'jpg', 'png'). Default is 'jpg'.

        Returns
        -------
        None
            This function generates a map with seismic events but does not 
            return any data.
        """
        self.mp.fig, self.mp.ax = self.mp.create_base_map(bounds_res, bmap_res)
        main_extent = self.mp.extent(self.ct.data, xlim=xlim, ylim=ylim)

        if isinstance(size, (int, float)):
            plt_size = size
        elif isinstance(size, str):
            plt_size = (self.ct.data[size]*size_scale_factor[0]) ** size_scale_factor[1]
        else:
            raise ValueError("The 'size' parameter must be a scalar or a column from your data.")

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
                x=large_quakes.lon, y=large_quakes.lat, c=hl_color, s=hl_size, marker=hl_marker, 
                edgecolor=hl_edgecolor, linewidth=0.75, label=f'Events M > {hl_ms}'
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
                    f"{'M' if size == 'mag' else 'D' if size == 'depth' else size} {v}" for v in size_values
                ]
                
                size_handles = [
                    plt.scatter([], [], s=(v*size_scale_factor[0]) ** size_scale_factor[1], 
                                facecolor='white', edgecolor='black', alpha=alpha, label=label)
                    for v, label in zip(size_values, size_legend_labels)
                ]
                
                leg2 = plt.legend(
                    handles=size_handles,
                    loc=size_legend_loc,
                    fancybox=False,
                    edgecolor='black',
                    ncol=len(size_values),
                    borderpad=1.2,
                )
                
                self.mp.ax.add_artist(leg2)

        if inset:
            self.mp.inset(main_extent, buffer=inset_buffer, bounds_res=bounds_res)

        if save_figure:
            self.bp.save_figure(save_name, save_extension)

        plt.show()

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
            save_name: str = 'map', 
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plots seismic event magnitudes over time.

        Parameters
        ----------
        size : float or str, optional
            The size of the markers representing seismic events. If a string 
            is provided, it should refer to a column in the DataFrame (e.g., 
            'mag') to scale point sizes proportionally. Default is 10.

        size_scale_factor : tuple[float, float], optional
            A tuple to scale marker sizes when `size` is based on a DataFrame 
            column. The first element scales the values, and the second element 
            raises them to a power. Default is (1, 3).

        color : str, optional
            The color used to fill the seismic event markers. Default is 
            'grey'.

        edgecolor : str, optional
            The color used for the edges of the seismic event markers. 
            Default is 'black'.

        alpha : float, optional
            The transparency level of the markers. A value between 0 and 
            1, where 1 is fully opaque and 0 is fully transparent. 
            Default is 0.75.

        size_legend : bool, optional
            If True, displays a legend that explains marker sizes. Default is False.
            
        size_legend_loc : str, optional
            Location of the size legend when `size_legend` is True. 
            Default is 'upper right'.

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is (10, 5).

        save_figure : bool, optional
            If set to True, the function saves the generated plots using 
            the provided base name and file extension. The default is False.

        save_name : str, optional
            The base name used for saving figures when `save_figure` is 
            True. It serves as the prefix for file names. The default base 
            name is 'section'.

        save_extension : str, optional
            The file extension to use when saving figures, such as 'jpg', 
            'png', etc. The default extension is 'jpg'.
        """
        self.bp.set_style(styling.DEFAULT)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title('Magnitude-time distribution', fontweight='bold')
        
        if isinstance(size, (int, float)):
            plt_size = size
        elif isinstance(size, str):
            plt_size = (self.ct.data[size]*size_scale_factor[0]) ** size_scale_factor[1]
        else:
            raise ValueError("The 'size' parameter must be a scalar or a column from your data.")

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
                    f"{'M' if size == 'mag' else 'D' if size == 'depth' else size} {v}" for v in size_values
                ]
                
                size_handles = [
                    plt.scatter([], [], s=(v*size_scale_factor[0]) ** size_scale_factor[1], 
                                facecolor='white', edgecolor='black', alpha=alpha, label=label)
                    for v, label in zip(size_values, size_legend_labels)
                ]
                
                plt.legend(
                    handles=size_handles,
                    loc=size_legend_loc,
                    fancybox=False,
                    edgecolor='black',
                    ncol=len(size_values),
                )

        ax.set_ylabel('Magnitude')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30, ha='right')

        ax.grid(True, alpha=0.25, axis='y', linestyle=':')

        if save_figure:
            self.bp.save_figure(save_name, save_extension)
        
        plt.show()
        self.bp.reset_style()

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
            save_name: str = 'map', 
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
            Column name used to color points by a specific attribute. If None, 
            uses a fixed color.

        cmap : str, optional
            Colormap to use when coloring points by an attribute. Default is 'jet'.

        hl_ms : int, optional
            Magnitude threshold for highlighting large seismic events. Default is None.

        hl_size : float, optional
            The size of the highlighted events. Default is 200.

        hl_marker : str, optional
            Marker style for highlighted events. Default is '*'.

        hl_color : str, optional
            Color for highlighted seismic events. Default is 'red'.

        hl_edgecolor : str, optional
            Edge color for highlighted events. Default is 'darkred'.

        size : float or str, optional
            Size of the points or the name of the column to use for size scaling. 
            Default is 10.

        size_scale_factor : tuple[float, float], optional
            Scaling factors (base, exponent) for the point sizes. Default is (1, 2).

        color : str, optional
            Default color for the points if `color_by` is None. Default is 'grey'.

        edgecolor : str, optional
            Color for the edges of the points. Default is 'black'.

        alpha : float, optional
            Transparency level of the points. Default is 0.75.

        xlim : tuple of str, optional
            Time limits for the x-axis as start and end date strings. Default is None.

        ylim : tuple[float, float], optional
            Limits for the y-axis (distance from center). Default is None.

        legend : str, optional
            Label for the points. Default is None.

        legend_loc : str, optional
            Location for the legend. Default is 'lower right'.

        size_legend : bool, optional
            If True, includes a legend for point sizes. Default is False.

        size_legend_loc : str, optional
            Location for the size legend. Default is 'upper right'.

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is (10, 5).

        save_figure : bool, optional
            If True, saves the figure. Default is False.

        save_name : str, optional
            Base name for the saved figure. Default is 'map'.

        save_extension : str, optional
            File extension for the saved figure. Default is 'jpg'.
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
                    plt.scatter([], [], s=(v * size_scale_factor[0]) ** size_scale_factor[1], 
                                facecolor='white', edgecolor='black', alpha=alpha, label=label)
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
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30, ha='right')

        ax.grid(True, alpha=0.25, axis='y', linestyle=':')

        if save_figure:
            self.bp.save_figure(save_name, save_extension)

        plt.show()
        self.bp.reset_style()

    def plot_event_timeline(
            self,
            fig_size: tuple[float, float] = (10, 5), 
            save_figure: bool = False,
            save_name: str = 'map', 
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plots a timeline of seismic events to visualize the cumulative 
        number of events over time.
        
        Parameters
        ----------
        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is (10, 5).

        save_figure : bool, optional
            If set to True, the function saves the generated plots using 
            the provided base name and file extension. The default is False.

        save_name : str, optional
            The base name used for saving figures when `save_figure` is 
            True. It serves as the prefix for file names. The default base 
            name is 'section'.

        save_extension : str, optional
            The file extension to use when saving figures, such as 'jpg', 
            'png', etc. The default extension is 'jpg'.
        """
        self.bp.set_style(styling.DEFAULT)
        
        events_sorted = self.ct.data.sort_values('time')
        time_data = pd.to_datetime(events_sorted['time'])

        fig = plt.figure(figsize=fig_size)
        plt.title('Event timeline', fontweight='bold')
        plt.plot(time_data, np.arange(len(events_sorted)), color='black')

        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30, ha='right')

        plt.ylabel('Cumulative no. of events')
        plt.ylim(0)
        plt.grid(True, axis='x', linestyle=':', linewidth=0.25)
        plt.tight_layout()
        
        if save_figure:
            self.bp.save_figure(save_name, save_extension)
        
        plt.show()
        self.bp.reset_style()

    def plot_attribute_distributions(
            self,
            fig_size: tuple[float, float] = (10, 6),
            save_figure: bool = False,
            save_name: str = 'map', 
            save_extension: str = 'jpg'
    ) -> None:
        """
        Visualizes the distribution of key attributes in the seismic event 
        catalog.
        
        Parameters
        ----------
        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is (10, 6).

        save_figure : bool, optional
            If set to True, the function saves the generated plots using 
            the provided base name and file extension. The default is False.

        save_name : str, optional
            The base name used for saving figures when `save_figure` is 
            True. It serves as the prefix for file names. The default base 
            name is 'map'.

        save_extension : str, optional
            The file extension to use when saving figures, such as 'jpg', 
            'png', etc. The default extension is 'jpg'.
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

        if save_figure:
            self.bp.save_figure(save_name, save_extension)
        
        plt.tight_layout()
        plt.show()
        self.bp.reset_style()


class SubCatalogPlotter:
    def __init__(self, sub_catalog: type) -> None:
            self.sc = sub_catalog
            self.mp = MapPlotter()
            self.bp = BasePlotter()
    
    def plot_on_section(self, normalize: bool = True):
        if self.sc.selected_from != 'CrossSection':
            raise ValueError('To be plotted on-section, the SubCatalog must be '
                             'sampled from a CrossSection object.')
        
        plt.figure(figsize=(12, 6))
        plt.scatter