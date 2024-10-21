import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
        size: float = 0.5, 
        color: str = 'lightgrey',
        edgecolor: str = 'grey', 
        alpha: float = 0.5, 
        legend: str = None, 
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
            Text for the legend describing the plotted seismic events. 
            If None, no legend is displayed.

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

        if color_by:
            if color_by not in self.ct.data.columns:
                raise ValueError(f"Column '{color_by}' not found in catalog data.")
            
            if color_by == 'time':
                global_min = mdates.date2num(self.ct.data[color_by].min())
                global_max = mdates.date2num(self.ct.data[color_by].max())
            else:
                global_min = np.floor(self.ct.data[color_by].min())
                global_max = np.ceil(self.ct.data[color_by].max())

            color = self.ct.data[color_by]

            if color_by == 'mag':
                colorbar_label = 'Magnitude'
            elif color_by == 'time':
                colorbar_label = 'Origin time'
            elif color_by == 'depth':
                colorbar_label = 'Depth [km]'
            else:
                colorbar_label = color_by

            if color_by == 'time':
                color_numeric = mdates.date2num(color)
                scatter = self.mp.scatter(
                    x=self.ct.data.lon, y=self.ct.data.lat, c=color_numeric, s=size, 
                    cmap=cmap, edgecolor=edgecolor, linewidth=0.25, alpha=alpha, 
                    label=legend, vmin=global_min, vmax=global_max
                )

                cbar = plt.colorbar(scatter, ax=self.mp.ax, orientation='horizontal', 
                                    pad=0.07, shrink=0.6, aspect=40)
                cbar.set_label(colorbar_label)
                cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(cbar.ax.get_xticklabels(), rotation=45, ha='right')

            else:
                scatter = self.mp.scatter(
                    x=self.ct.data.lon, y=self.ct.data.lat, c=color, s=size, cmap=cmap, 
                    edgecolor=edgecolor, linewidth=0.25, alpha=alpha, label=legend,
                    vmin=global_min, vmax=global_max
                )
                cbar = plt.colorbar(scatter, ax=self.mp.ax, orientation='horizontal', 
                                    pad=0.07, shrink=0.6, aspect=40)
                cbar.set_label(colorbar_label)
        else:
            self.mp.scatter(
                x=self.ct.data.lon, y=self.ct.data.lat, c=color, s=size, 
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
            leg = plt.legend(loc='lower left', fancybox=False, edgecolor='black')
            leg.legend_handles[0].set_sizes([50])
            leg.legend_handles[1].set_sizes([70])

        if inset:
            self.mp.inset(main_extent, buffer=inset_buffer, bounds_res=bounds_res)

        if save_figure:
            self.mp.save_figure(save_name, save_extension)

        plt.show()

    def plot_magnitude_time(
        self,
        save_figure: bool = False,
        save_name: str = 'map', 
        save_extension: str = 'jpg'
    ) -> None:
        """
        Plots seismic event magnitudes over time.

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
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title('Magnitude-time distribution', fontweight='bold')
        
        ax.scatter(
            self.ct.data.time, self.ct.data.mag, c='lightgrey', 
            s=self.ct.data.mag, edgecolor='grey', linewidth=0.25, 
            alpha=0.5, zorder=10
        )
        ax.set_ylabel('Magnitude')

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30, ha='right')

        ax.grid(True, alpha=0.25, axis='y', linestyle=':')
        
        if save_figure:
            self.bp.save_figure(fig, save_name, save_extension)
        
        plt.show()

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
            self.bp.save_figure(fig, save_name, save_extension)
        
        plt.show()

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
            self.bp.save_figure(fig, save_name, save_extension)
        
        plt.tight_layout()
        plt.show()