import srtm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.patches as patches

from seismoviz.plotters.common import styling
from seismoviz.utils import convert_to_geographical
from seismoviz.plotters.common.map_plotter import MapPlotter
from seismoviz.plotters.common.base_plotter import BasePlotter

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


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
            hl_ms: float = None,
            hl_size: float = 200,
            hl_marker: str = '*',
            hl_color: str = 'red',
            hl_edgecolor: str = 'darkred',
            color: str = 'black',
            edgecolor: str = None,
            size: float | str = 5,
            size_scale_factor: tuple[float, float] = (1, 2),
            alpha: float = 0.5,
            legend: str = None,
            legend_loc: str = 'lower left',
            size_legend: bool = False,
            size_legend_loc: str = 'upper right',
            scale_legend: bool = True,
            scale_legend_loc: str  = 'lower right',
            ylabel: str = 'Depth [km]',
            xlim: tuple[float, float] = None,
            ylim: tuple[float, float] = None,
            fig_size: tuple[float, float] = (12, 6),
            facecolor: tuple[str, str] = ('#F0F0F0', '#FFFFFF'),
            save_figure: bool = False,
            save_name: str = 'cross_section',
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plots a cross-section of seismic events with customizable appearance.

        Parameters
        ----------
        color_by : str, optional
            Column name from the DataFrame to color seismic events (e.g., 
            'magnitude', 'depth'). Default is None, using a uniform color.

        cmap : str, optional
            Colormap to use when `color_by` is specified. Default is 'jet'.

        title : str, optional
            Title of the plot. Default is 'Section'.

        hl_ms : float, optional
            If specified, highlights seismic events with a magnitude greater 
            than this value using special markers. Default is None.

        hl_size : float, optional
            Size of the markers used for highlighting seismic events (when 
            `hl_ms` is specified). Default is 200.

        hl_marker : str, optional
            Marker style for highlighted events. Default is '*'.

        hl_color : str, optional
            Color for highlighted seismic event markers. Default is 'red'.

        hl_edgecolor : str, optional
            Edge color for highlighted event markers. Default is 'darkred'.

        color : str, optional
            Default color for seismic event markers. Default is 'black'.

        edgecolor : str, optional
            Edge color for event markers. If None, the default edge color is 
            not applied. Default is None.

        size : float or str, optional
            Size of markers representing seismic events. If a string is 
            provided, it should refer to a column from the DataFrame (e.g., 
            'magnitude') to scale marker sizes proportionally. Default is 5.

        size_scale_factor : tuple[float, float], optional
            Tuple for scaling marker sizes when `size` is based on a column 
            in the DataFrame. The first element scales values, and the second 
            raises them to a power. Default is (1, 2).

        alpha : float, optional
            Transparency level for markers, where 0 is fully transparent and 
            1 is fully opaque. Default is 0.5.

        legend : str, optional
            Label for the legend describing the seismic events. Default is None.

        legend_loc : str, optional
            Location of the main legend on the plot. Default is 'lower left'.

        size_legend : bool, optional
            If True, displays a legend that explains marker sizes. Default is True.

        size_legend_loc : str, optional
            Location of the size legend if it is shown. Default is 'upper right'.

        scale_legend: bool, optional
            If True, displays a scale bar on the plot to indicate real-world distances.
            Default is True.

        scale_legend_loc : str, optional
                Location of the scale legend (e.g., size scaling). If False, the 
                scale legend is not displayed. Default is 'lower right'.

        ylabel : str, optional
            Label for the y-axis (typically indicating depth). Default is 'Depth [km]'.

        xlim : tuple[float, float], optional
            Horizontal extent (limits) of the plot (x-axis). If None, limits 
            are set automatically. Default is None.

        ylim : tuple[float, float], optional
            Vertical extent (limits) of the plot (y-axis). If None, limits 
            are set automatically. Default is None.

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is (12, 6).

        facecolor : tuple[str, str], optional
            Tuple specifying the background colors of the plot, with two 
            values for gradient-like effects. Default is ('#F0F0F0', '#FFFFFF').

        save_figure : bool, optional
            If True, saves the plot to a file. Default is False.

        save_name : str, optional
            Base name of the file to save when `save_figure` is True. 
            Default is 'cross_section'.

        save_extension : str, optional
            File format to use when saving the plot (e.g., 'jpg', 'png'). 
            Default is 'jpg'.

        Returns
        -------
        None
            This function generates a cross-section plot but does not return 
            any data.
        """
        elev_profiles = self._get_elevation_profiles()

        for section in range(self.cs.data.index.get_level_values('section_id').nunique()):
            self.bp.set_style(styling.CROSS_SECTION)
            fig, ax = plt.subplots(figsize=fig_size)

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

            if isinstance(size, (int, float)):
                plt_size = size
            elif isinstance(size, str):
                plt_size = (self.cs.data.loc[section][size] * size_scale_factor[0]) ** size_scale_factor[1]
            else:
                raise ValueError("The 'size' parameter must be a scalar or a column from your data.")

            if color_by:
                fig.set_figheight(8)
                self.bp.plot_with_colorbar(
                    ax=ax,
                    data=self.cs.data.loc[section],
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
                    self.cs.data.loc[section].on_section_coords,
                    self.cs.data.loc[section].depth,
                    color=color, 
                    edgecolor=edgecolor,
                    s=plt_size, 
                    alpha=alpha,
                    linewidth=0.25,
                    label=legend
                )

            if hl_ms is not None:
                large_quakes = self.cs.data.loc[section][self.cs.data.loc[section]['mag'] > hl_ms]
                ax.scatter(
                    x=large_quakes.on_section_coords, y=large_quakes.depth, c=hl_color, s=hl_size,
                    marker=hl_marker, edgecolor=hl_edgecolor, linewidth=0.75,
                    label=f'Events M > {hl_ms}'
                )

            ax.set_title(f'{title} {section + 1}', fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.xaxis.set_visible(False)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.25, linestyle=':')

            if scale_legend:
                scale_length = (self.cs.map_length / 2) / 5
                scale_label = f'{scale_length:.1f} km'

                scalebar = AnchoredSizeBar(
                    transform=ax.transData,
                    size=scale_length,
                    label=scale_label,
                    loc=scale_legend_loc,
                    sep=5,
                    color='black',
                    frameon=False,
                    size_vertical=0.1,
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
                    min_size = np.floor(min(self.cs.data.loc[section][size]))
                    max_size = np.ceil(max(self.cs.data.loc[section][size]))
                    size_values = [min_size, (min_size + max_size) / 2, max_size]
                    size_legend_labels = [
                        f"{'M' if size == 'mag' else 'D' if size == 'depth' else size} {v}" for v in size_values
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
                    )

                    ax.add_artist(leg2)

            if save_figure:
                self.bp.save_figure(save_name, save_extension)

            plt.show()
            self.bp.reset_style()
    
    def plot_section_lines(
            self,
            title: str = None,
            hl_ms: int = None,
            hl_size: int = 300,
            hl_marker: str = '*',
            hl_color: str = 'red',
            hl_edgecolor: str = 'darkred',
            size: float = 10,
            color: str = 'grey',
            edgecolor: str = None, 
            alpha: float = 0.75, 
            legend: str = None,
            legend_loc: str = 'lower left',
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
        title : str, optional
            Title of the map. If None, no title is displayed. Default is None.

        hl_ms : int, optional
            If specified, highlights seismic events with a magnitude 
            greater than this value using different markers. Default is None.

        hl_size : float, optional
            Size of the markers used for highlighting seismic events (when 
            `hl_ms` is specified). Default is 200.

        hl_marker : str, optional
            Marker style for highlighted events. Default is '*'.

        hl_color : str, optional
            Color for highlighted seismic event markers. Default is 'red'.

        hl_edgecolor : str, optional
            Edge color for highlighted event markers. Default is 'darkred'.

        size : float or str, optional
            The size of the markers representing seismic events. If a string 
            is provided, it should refer to a column in the DataFrame (e.g., 
            'magnitude') to scale point sizes proportionally. Default is 10.

        color : str, optional
            Default color for event markers when `color_by` is None. 
            Default is 'grey'.

        edgecolor : str, optional
            Edge color for event markers. Default is None.

        alpha : float, optional
            Transparency level for markers, ranging from 0 (transparent) to 1 
            (opaque). Default is 0.75.

        legend : str, optional
            Text for the legend describing the seismic events. If None, 
            no legend is displayed. Default is None.

        legend_loc : str, optional
            Location of the legend for the seismic event markers. 
            Default is 'lower left'.

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

        if hl_ms is not None:
            large_quakes = self.cs.data[self.cs.data['mag'] > hl_ms]
            self.mp.scatter(
                x=large_quakes.lon, y=large_quakes.lat, c=hl_color, s=hl_size,
                marker=hl_marker, edgecolor=hl_edgecolor, linewidth=0.75,
                label=f'Events M > {hl_ms}'
            )

        for idx, (lons, lats) in enumerate(self._get_section_lines()):
            self.mp.plot(lons, lats, color='black', linewidth=1)
            self.mp.annotate(
                text=str(idx + 1),
                xy=(lons[1], lats[1]),
                ha='center',
                va='center',
                fontsize=10,
                color='white',
                weight='bold',
                bbox=dict(
                    boxstyle='circle',
                    facecolor='black',
                    edgecolor='black',
                    pad=0.2)
            )

        main_extent = self.mp.extent({'lon': lon, 'lat': lat}, xlim=xlim, ylim=ylim)

        if legend:
            leg = plt.legend(loc=legend_loc, fancybox=False, edgecolor='black')
            leg.legend_handles[0].set_sizes([50])
            
            if hl_ms is not None:
                leg.legend_handles[1].set_sizes([90])

        if inset:
            self.mp.inset(main_extent, buffer=inset_buffer, bounds_res=bounds_res)

        if save_figure:
            self.bp.save_figure(save_name, save_extension)

        plt.show()