import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import seismoviz.components.visualization.utils.plot_utils as pu

from seismoviz.components.visualization.utils import styling
from seismoviz.components.visualization.plot_common import CommonPlotter


class SubCatalogPlotter(CommonPlotter):
    """
    Provides plotting methods for SubCatalog objects.
    """

    def __init__(self, sub_catalog: type) -> None:
        super().__init__(sub_catalog.data)
        self.sc = sub_catalog

    def plot_on_section(
            self, 
            title: str = None,
            color_by: str = None,
            cmap: str = 'jet',
            hl_ms: float = None,
            hl_size: float = 200,
            hl_marker: str = '*',
            hl_color: str = 'red',
            hl_edgecolor: str = 'darkred',
            size: float | str = 1,
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
            normalize: bool = True,
            save_figure: bool = False,
            save_name: str = 'on_section',
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plots seismic events on a cross-sectional view, typically after being
        selected from a CrossSection object.

        Parameters
        ----------
        title : str, optional
            Title of the plot. If ``None``, no title is displayed. Default is ``None``.
    
        color_by : str, optional
            Specifies the column in the DataFrame used to color the
            seismic events. Default is ``None``, which applies a single color to
            all points.
    
        cmap : str, optional
            The colormap to use for coloring events if ``color_by`` is specified.
            Default is ``'jet'``.
    
        hl_ms : float, optional
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
            point sizes proportionally. Default is 1.
    
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
            is ``'upper right'``.
    
        scale_legend: bool, optional
            If ``True``, displays a legend that shows a scale bar on the plot to
            indicate real-world distances. Default is ``True``.
    
        scale_legend_loc : str, optional
            Location of the scale legend when ``scale_legend`` is ``True``.
            Default is ``'lower right'``.
    
        ylabel : str, optional
            Label for the y-axis. Default is ``'Depth [km]'``.
    
        normalize : bool, optional
            If ``True``, normalizes the on-section coordinates by subtracting the median.
            Default is ``True``.
    
        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.
    
        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is
            ``'on_section'``.
    
        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``).
            Default is ``'jpg'``.
    
        Returns
        -------
        None
            A cross-sectional plot of seismic events.

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

            # Create a SubCatalog by selecting events from the cross-section
            sub_catalog = cs.select_events(criteria='depth > 5')

            # Plot the events on the cross-section
            sub_catalog.plot_on_section(
                color_by='mag',
                cmap='viridis',
                size='mag',
                legend='Selected Events',
                title='Events with Depth > 5 km'
            )
        """
        if self.sc.selected_from != 'CrossSection':
            raise ValueError('To be plotted on-section, the SubCatalog must be '
                             'sampled from a CrossSection object.')
        pu.set_style(styling.DEFAULT)
        
        osc = self.sc.data.on_section_coords.copy()
        if normalize:
            osc = osc - osc.median()
            self.sc.data['on_section_coords_normalized'] = osc
            x_coords = 'on_section_coords_normalized'
        else:
            x_coords = 'on_section_coords'
        
        # Process size parameter
        plt_size = pu.process_size_parameter(size, self.sc.data, size_scale_factor)

        fig, ax = plt.subplots(figsize=(12, 6))

        if color_by:
            fig.set_figheight(8)
            pu.plot_with_colorbar(
                ax=ax,
                data=self.sc.data,
                x=x_coords,
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
                self.sc.data[x_coords],
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
            pu.plot_highlighted_events(
                ax=ax,
                data=self.sc.data,
                hl_ms=hl_ms,
                hl_size=hl_size,
                hl_marker=hl_marker,
                hl_color=hl_color,
                hl_edgecolor=hl_edgecolor,
                x=x_coords,
                y='depth'
            )
        
        ax.set_ylabel(ylabel)
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
                leg.legend_handles[-1].set_sizes([90])

            ax.add_artist(leg)

            if isinstance(size, str) and size_legend:
                pu.create_size_legend(
                    ax=ax,
                    size=size,
                    data=self.sc.data,
                    size_scale_factor=size_scale_factor,
                    alpha=alpha,
                    size_legend_loc=size_legend_loc
                )
        
        if save_figure:
            pu.save_figure(save_name, save_extension)
        
        plt.show()
        pu.reset_style()
