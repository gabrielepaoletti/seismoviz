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
        highlight_mag: int = None, 
        color_by: str = None, 
        cmap: str = 'jet', 
        title: str = None, 
        size: float = 10,
        size_scale_factor: tuple[int, int] = (1, 3),
        color: str = 'grey',
        edgecolor: str = 'black', 
        alpha: float = 0.75, 
        legend: str = None,
        scale_legend: bool = True,
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
        Visualizes seismic events on a map.

        Parameters
        ----------
        highlight_mag : int, optional
            If specified, highlights all seismic events with a magnitude 
            greater than this value by plotting them as stars.

        color_by : str, optional
            Specifies the column in the DataFrame used to color the 
            seismic events. Common options include 'magnitude', 'time', 
            or 'depth'. If not provided, a default color is used.

        cmap : str, optional
            The colormap to use when coloring the events based on the 
            `color_by` column. Default is 'jet'.

        title : str, optional
            The title to be displayed above the map. If not provided, 
            the map will have no title.

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

        legend : str, optional
            Text for the legend describing the plotted seismic events. 
            If None, no legend is displayed.
        
        scale_legend : bool, optional
            If True, displays a legend for the point sizes, indicating how 
            they correspond to specific values. Default is True.

        xlim : tuple[float, float], optional
            A tuple specifying the minimum and maximum longitude values 
            to set the map extent horizontally. If not provided, the 
            extent will be set automatically based on the data.

        ylim : tuple[float, float], optional
            A tuple specifying the minimum and maximum latitude values 
            to set the map extent vertically. If not provided, the 
            extent will be set automatically based on the data.

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
            the provided base name and file extension. The default is False.

        save_name : str, optional
            The base name used for saving figures when `save_figure` is 
            True. It serves as the prefix for file names. The default 
            base name is 'map'.

        save_extension : str, optional
            The file extension to use when saving figures, such as 'jpg', 
            'png', etc. The default extension is 'jpg'.
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

        if highlight_mag is not None:
            large_quakes = self.ct.data[self.ct.data['mag'] > highlight_mag]
            self.mp.scatter(
                x=large_quakes.lon, y=large_quakes.lat, c='red', s=200, marker='*', 
                edgecolor='darkred', linewidth=0.75, label=f'Events M > {highlight_mag}'
            )

        if legend:
            leg = plt.legend(
                loc='lower left',
                fancybox=False,
                edgecolor='black'
            )
            leg.legend_handles[0].set_sizes([50])
            leg.legend_handles[1].set_sizes([90])
            self.mp.ax.add_artist(leg)
            
            if isinstance(size, str) and scale_legend:
                min_size = np.floor(min(self.ct.data[size]))
                max_size = np.ceil(max(self.ct.data[size]))
                size_values = [min_size, (min_size + max_size)/2, max_size]
                size_legend_labels = [f"{'M' if size == 'mag' else 'D' if size == 'depth' else size} {v}" for v in size_values]
                
                size_handles = [
                    plt.scatter([], [], s=(v*size_scale_factor[0]) ** size_scale_factor[1], 
                                facecolor='white', edgecolor='black', alpha=alpha, label=label)
                    for v, label in zip(size_values, size_legend_labels)
                ]
                
                leg2 = plt.legend(
                    handles=size_handles,
                    loc='lower right',
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
        self, save_figure: bool = False, save_name: str = 'map', 
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