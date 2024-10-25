import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from seismoviz.plotters.common import styling
from seismoviz.plotters.common.map_plotter import MapPlotter
from seismoviz.plotters.common.base_plotter import BasePlotter


class CatalogPlotter:
    def __init__(self, catalog: type) -> None:
        self.ct = catalog
        self.mp = MapPlotter()
        self.bp = BasePlotter()

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
            'magnitude') to scale point sizes proportionally. Default is 10.

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
        save_figure: bool = False,
        save_name: str = 'map', 
        save_extension: str = 'jpg'
    ) -> None:
        """
        Plots seismic event magnitudes over time.

        Parameters
        ----------
        size : float | str, optional
            The size of the markers used to represent seismic events on 
            the map. Default is 10.

            .. note::
                If you want to plot events where the point size is proportional
                to a specific dimension (e.g., magnitude or depth), you can
                directly pass the corresponding column from the `pd.DataFrame`
                to the argument as a string (`size='mag'`).

        size_scale_factor : tuple[float, float], optional
            A tuple of two factors used to scale the size of the markers when `size` is 
            based on a column from the data. The size is calculated by first multiplying 
            the values in the specified column by the first element of the tuple 
            (`size_scale_factor[0]`), and then raising the result to the power of the 
            second element (`size_scale_factor[1]`). Default is (1, 2).

            .. note::
                For example, if `size='mag'`, the size of the markers is calculated as:
                `plt_size = (magnitude * size_scale_factor[0]) ** size_scale_factor[1]`.

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

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title('Magnitude-time distribution', fontweight='bold')
        
        if isinstance(size, (int, float)):
            plt_size = size
        elif isinstance(size, str):
            plt_size = (self.ct.data[size]*2) ** size_scale_factor
        else:
            raise ValueError("The 'size' parameter must be a scalar or a column from your data.")

        if color_by:
            fig.set_figwidth(12)
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
        ax.set_ylabel('Magnitude')

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
        save_figure: bool = False,
        save_name: str = 'map', 
        save_extension: str = 'jpg'
    ) -> None:
        """
        Plots a timeline of seismic events to visualize the cumulative 
        number of events over time.
        
        Parameters
        ----------
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

        fig = plt.figure(figsize=(10, 5))
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
        save_figure: bool = False,
        save_name: str = 'map', 
        save_extension: str = 'jpg'
    ) -> None:
        """
        Visualizes the distribution of key attributes in the seismic event 
        catalog.
        
        Parameters
        ----------
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
        fig, ax = plt.subplots(rows, cols, figsize=(10, 6), sharey=True)
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