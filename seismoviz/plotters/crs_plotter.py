import srtm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.patches as patches

from seismoviz.utils import convert_to_geographical
from seismoviz.plotters.common.map_plotter import MapPlotter
from seismoviz.plotters.common.base_plotter import BasePlotter


class CrossSectionPlotter:
    def __init__(self, cross_section: type) -> None:
        self.cs = cross_section
        self.mp = MapPlotter()
        self.bp = BasePlotter()

    def _get_section_lines(self) -> list[tuple[list[float], list[float]]]:
        """
        Calculates and returns the geographical coordinates (latitude and longitude) 
        for each section line.

        Returns
        -------
        list[tuple[list[float], list[float]]]
            List of section lines, where each line contains two lists: one for 
            longitudes and one for latitudes.
        """
        angle = np.radians(90 - (self.cs.strike - 90))

        delta_x = self.cs.map_length / 2 * np.cos(angle)
        delta_y = self.cs.map_length / 2 * np.sin(angle)

        section_lines = []

        for center in self.cs._center_coords:
            center_x, center_y, _ = center

            point1_x, point1_y = center_x - delta_x, center_y - delta_y
            point2_x, point2_y = center_x + delta_x, center_y + delta_y

            lons, lats = convert_to_geographical(
                utmx=[point1_x, point2_x],
                utmy=[point1_y, point2_y],
                zone=self.cs.zone,
                northern=True if self.cs.hemisphere == 'north' else False,
                units='km'
            )

            section_lines.append((lons, lats))

        return section_lines

    def _get_elevation_profiles(self) -> list[list[float]]:
        """
        Calculates the elevation profiles for all section lines.

        Returns
        -------
        list[list[float]]
            A list of lists, where each inner list contains the elevation profile 
            for a specific section line.
        """
        elevation_data = srtm.get_data()

        all_elevations = []
        for lons, lats in self._get_section_lines():
            lons_resampled = np.linspace(lons[0], lons[-1], 250)
            lats_resampled = np.linspace(lats[0], lats[-1], 250)

            elevations = []
            for lon, lat in zip(lons_resampled, lats_resampled):
                elevation = elevation_data.get_elevation(lat, lon)
                if elevation is None:
                    elevation = elevations[-1] if elevations else 0
                elevations.append(elevation)

            all_elevations.append(elevations)

        return all_elevations

    def plot_sections(
        self,
        color_by: str = None,
        cmap: str = 'jet',
        title: str = 'Section',
        color: str = 'black',
        edgecolor: str = None,
        size: float = 5,
        alpha: float = 0.5,
        ylabel: str = 'Depth [km]',
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        scale_loc: str | bool = 'lower right',
        facecolor: tuple[str, str] = ('#F0F0F0', '#FFFFFF'),
        save_figure: bool = False,
        save_name: str = 'cross_section',
        save_extension: str = 'jpg'
    ) -> None:
        """
        Visualizes seismic events on section.

        Parameters
        ----------
        color_by : str, optional
            Specifies the column in the DataFrame used to color the seismic events. 
            Common options include 'magnitude', 'time', or 'depth'. If not provided, 
            a default color is used.

        cmap : str, optional
            The colormap to use when coloring the events based on the `color_by` 
            column. Default is 'jet'.

        title : str, optional
            The title displayed at the top of the plot. Defaults to 'Section'.

        size : float, optional
            The size of the markers used to represent seismic events on the map. 
            Default is 5.

        color : str, optional
            The color used to fill the seismic event markers. Default is 'black'.

        edgecolor : str, optional
            The color used for the edges of the seismic event markers. Default is 
            None.

        alpha : float, optional
            The transparency level of the markers. A value between 0 and 1, where 
            1 is fully opaque and 0 is fully transparent. Default is 0.5.

        ylabel : str, optional
            Label for the y-axis, usually indicating depth. Defaults to 'Depth [km]'.

        xlim : tuple[float, float], optional
            Limits for the x-axis as a tuple (min, max). If None, the limits are 
            determined based on the data. Defaults to None.

        ylim : tuple[float, float], optional
            Limits for the y-axis as a tuple (min, max), controlling the depth range 
            displayed. If None, the limits are auto-scaled based on the data. 
            Defaults to None.

        scale_loc : str or bool, optional
            Location for the scale bar (e.g., 'upper right', 'lower left', etc.). 
            If set to False, no scale bar is shown. Defaults to 'lower right'.

        facecolor : tuple[str, str], optional
            Background color for the plot, specified as a hex code or recognized 
            color name. Defaults to a light grey ('#F0F0F0').

        save_figure : bool, optional
            If set to True, the function saves the generated plots using the 
            provided base name and file extension. The default is False.

        save_name : str, optional
            The base name used for saving figures when `save_figure` is True. It 
            serves as the prefix for file names. The default base name is 
            'cross_section'.

        save_extension : str, optional
            The file extension to use when saving figures, such as 'jpg', 'png', 
            etc... The default extension is 'jpg'.

        Raises
        ------
        ValueError
            If scale_loc is not one of the valid location strings or False.
        """
        elev_profiles = self._get_elevation_profiles()

        for section in range(self.cs.data.index.get_level_values('section_id').nunique()):
            plt.rcParams['axes.spines.right'] = True
            fig, ax = plt.subplots(figsize=(12, 6))

            elev_profile = np.array(elev_profiles[section]) / 1000

            x_lim = (-self.cs.map_length / 2, self.cs.map_length / 2) if not xlim else xlim
            ax.set_xlim(*x_lim)

            y_lim = (-np.ceil(elev_profile.max()), self.cs.depth_range[1]) if not ylim else ylim
            ax.set_ylim(*y_lim)

            if facecolor:
                gradient = np.linspace(0, 1, 256).reshape(256, 1)
                fc_map = mcolors.LinearSegmentedColormap.from_list(
                    'gradient', [facecolor[0], facecolor[1]]
                )

                ax.imshow(
                    gradient, aspect='auto', cmap=fc_map,
                    extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]],
                    origin='upper', zorder=-10
                )

            ax.plot(
                np.linspace(-self.cs.map_length / 2, self.cs.map_length / 2, len(elev_profile)),
                -elev_profile, color='black', lw=1
            )

            if color_by:
                self.bp.plot_with_colorbar(
                    ax=ax,
                    data=self.cs.data.loc[section],
                    x='on_section_coords',
                    y='depth',
                    color_by=color_by,
                    cmap=cmap,
                    edgecolor=edgecolor,
                    size=size,
                    alpha=alpha
                )
            else:
                ax.scatter(
                    self.cs.data.loc[section].on_section_coords,
                    self.cs.data.loc[section].depth,
                    marker='.', color=color, edgecolor=edgecolor,
                    s=size, alpha=alpha, linewidth=0.25
                )

            ax.set_title(f'{title} {section + 1}', fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.xaxis.set_visible(False)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.25, linestyle=':')

            if scale_loc:
                scale_length = (self.cs.map_length / 2) / 5
                scale_label = f'{scale_length:.1f} km'

                if scale_loc == 'upper right':
                    scale_x_start = ax.get_xlim()[1] - (scale_length * 1.1)
                    scale_y = ax.get_ylim()[1] + (self.cs.depth_range[1] - self.cs.depth_range[0]) * 0.09
                elif scale_loc == 'upper left':
                    scale_x_start = ax.get_xlim()[0] + (scale_length * 0.1)
                    scale_y = ax.get_ylim()[1] + (self.cs.depth_range[1] - self.cs.depth_range[0]) * 0.09
                elif scale_loc == 'upper center':
                    scale_x_start = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2 - (scale_length / 2)
                    scale_y = ax.get_ylim()[1] + (self.cs.depth_range[1] - self.cs.depth_range[0]) * 0.09
                elif scale_loc == 'lower right':
                    scale_x_start = ax.get_xlim()[1] - (scale_length * 1.1)
                    scale_y = ax.get_ylim()[0] - (self.cs.depth_range[1] - self.cs.depth_range[0]) * 0.06
                elif scale_loc == 'lower left':
                    scale_x_start = ax.get_xlim()[0] + (scale_length * 0.1)
                    scale_y = ax.get_ylim()[0] - (self.cs.depth_range[1] - self.cs.depth_range[0]) * 0.06
                elif scale_loc == 'lower center':
                    scale_x_start = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2 - (scale_length / 2)
                    scale_y = ax.get_ylim()[0] - (self.cs.depth_range[1] - self.cs.depth_range[0]) * 0.06
                else:
                    valid_locations = [
                        'upper right', 'upper left', 'upper center',
                        'lower right', 'lower left', 'lower center'
                    ]
                    raise ValueError(f'Invalid value for scale_loc: {scale_loc}.'
                                     f'Valid options are {valid_locations}.')

                rect_width = scale_length / 4
                rect_height = (self.cs.depth_range[1] - self.cs.depth_range[0]) * 0.02

                for i in range(4):
                    rect_x = scale_x_start + i * rect_width
                    rec_col = 'black' if i % 2 == 0 else 'white'
                    rect = patches.Rectangle(
                        (rect_x, scale_y), rect_width, rect_height,
                        facecolor=rec_col, edgecolor='black'
                    )
                    ax.add_patch(rect)

                ax.text(
                    scale_x_start + scale_length / 2,
                    scale_y + rect_height * -0.5,
                    scale_label,
                    ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color='black'
                )

            if save_figure:
                self.bp.save_figure(fig, save_name, save_extension)

            plt.show()
    
    def plot_section_lines(
        self,
        highlight_mag: int = None,
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
        Visualizes section lines on a map.

        Parameters
        ----------
        highlight_mag : float, optional
            If specified, highlights all seismic events (that are present 
            in your sections) with a magnitude greater than this value 
            by plotting them as stars.

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

        xlim : tuple[float, float], optional
            A tuple specifying the minimum and maximum longitude values 
            to set the map extent horizontally. If not provided, the 
            extent will be set automatically based on the data.

        ylim : tuple[float, float], optional
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
        self.mp.fig, self.mp.ax = self.mp.create_base_map(bounds_res, bmap_res)

        lon, lat = convert_to_geographical(
            utmx=self.cs._utmx, utmy=self.cs._utmy, zone=self.cs.zone,
            northern=True if self.cs.hemisphere == 'north' else False,
            units='km'
        )
        self.mp.scatter(
            x=lon, y=lat, c=color, s=size, edgecolor=edgecolor,
            linewidth=0.25, alpha=alpha, label=legend
        )

        if title:
            self.mp.ax.set_title(title, fontweight='bold')

        if highlight_mag is not None:
            large_quakes = self.cs.data[self.cs.data['mag'] > highlight_mag]
            self.mp.scatter(
                x=large_quakes.lon, y=large_quakes.lat, c='red', s=200,
                marker='*', edgecolor='darkred', linewidth=0.75,
                label=f'Events M > {highlight_mag}'
            )

        for idx, (lons, lats) in enumerate(self._get_section_lines()):
            self.mp.plot(lons, lats, color='black', linewidth=1)
            self.mp.annotate(
                text=str(idx + 1), xy=(lons[1], lats[1]), ha='center',
                va='center', fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='circle', facecolor='black', edgecolor='black', pad=0.2)
            )

        main_extent = self.mp.extent({'lon': lon, 'lat': lat}, xlim=xlim, ylim=ylim)

        if legend:
            leg = plt.legend(loc='lower left', fancybox=False, edgecolor='black')
            leg.legend_handles[0].set_sizes([50])
            leg.legend_handles[1].set_sizes([70])

        if inset:
            self.mp.inset(main_extent, buffer=inset_buffer, bounds_res=bounds_res)

        if save_figure:
            self.mp.save_figure(save_name, save_extension)

        plt.show()