import srtm
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from seismoviz.utils import convert_to_geographical
from seismoviz.components.common import styling
from seismoviz.components.common.map_plotter import MapPlotter
from seismoviz.components.common.base_plotter import BasePlotter


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
                    size_vertical=(self.cs.data.loc[section].depth.max() - self.cs.data.loc[section].depth.min()) / 100,
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
                self.bp.save_figure(f'{save_name}_{section}', save_extension)

            plt.show()
            self.bp.reset_style()
    
    def plot_section_lines(
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
            sl_color: str = 'black',
            sl_linewidth: float = 1.5,
            sl_text_size: float = 10,
            sl_text_color: str = 'white',
            sl_text_weight: str = 'bold',
            sl_box_style: str = 'circle',
            sl_box_color: str = 'black',
            sl_box_edgecolor: str = 'black',
            sl_box_pad: float = 0.3,
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
        """
        self.mp.transform, self.mp.projection = transform, projection

        self.mp.fig, self.mp.ax = self.mp.create_base_map(
            terrain_style=terrain_style,
            terrain_cmap=terrain_cmap,
            terrain_alpha=terrain_alpha,
            bounds_res=bounds_res, 
            bmap_res=bmap_res
        )

        if isinstance(size, (int, float)):
            plt_size = size
        elif isinstance(size, str):
            plt_size = (self.cs.catalog[size]*size_scale_factor[0]) ** size_scale_factor[1]
        else:
            raise ValueError("The 'size' parameter must be a scalar or a column from your data.")
        import pandas as pd
        if color_by:
            self.mp.fig.set_figheight(10)
            self.mp.plot_with_colorbar(
                data=self.cs.catalog,
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
                x=self.cs.catalog.lon, y=self.cs.catalog.lat, c=color, s=plt_size, 
                edgecolor=edgecolor, linewidth=0.25, alpha=alpha, label=legend
            )
        main_extent = self.mp.extent(self.cs.catalog, xlim=xlim, ylim=ylim)

        if title:
            self.mp.ax.set_title(title, fontweight='bold')

        if hl_ms is not None:
            large_quakes = self.cs.catalog[self.cs.catalog['mag'] > hl_ms]
            self.mp.scatter(
                x=large_quakes.lon, y=large_quakes.lat, c=hl_color, s=hl_size, marker=hl_marker, 
                edgecolor=hl_edgecolor, linewidth=0.75, label=f'Events M > {hl_ms}'
            )

        for idx, (lons, lats) in enumerate(self._get_section_lines()):
            self.mp.plot(lons, lats, color=sl_color, linewidth=sl_linewidth)
            self.mp.annotate(
                text=str(idx + 1),
                xy=(lons[1], lats[1]),
                ha='center',
                va='center',
                fontsize=sl_text_size,
                color=sl_text_color,
                weight=sl_text_weight,
                bbox=dict(
                    boxstyle=sl_box_style,
                    facecolor=sl_box_color,
                    edgecolor=sl_box_edgecolor,
                    pad=sl_box_pad)
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
                min_size = np.floor(min(self.cs.data[size]))
                max_size = np.ceil(max(self.cs.data[size]))
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
                    ncol=len(size_values)
                )
                
                self.mp.ax.add_artist(leg2)

        if inset:
            self.mp.inset(main_extent, buffer=inset_buffer, bounds_res=bounds_res)

        if save_figure:
            self.bp.save_figure(save_name, save_extension)

        plt.show()