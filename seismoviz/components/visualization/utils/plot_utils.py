import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.axes import Axes
from matplotlib.collections import PathCollection


def set_style(style: dict) -> None:
    """
    Sets a custom style for plotting.
    """
    plt.rcParams.update(style)


def reset_style() -> None:
    """
    Resets plotting style to the default one.
    """
    plt.rcdefaults()


def format_x_axis_time(ax: Axes) -> None:
    """
    Formats the x-axis based on the displayed range of the axis.
    """
    x_min, x_max = mdates.num2date(ax.get_xlim())
    time_range = x_max - x_min
    _configure_time_axis(ax.xaxis, time_range)
    plt.xticks(rotation=30, ha='right')


def format_colorbar_time(cbar, orientation: str):
    """
    Formats the colorbar based on the displayed range and orientation.
    """
    x_min, x_max = mdates.num2date(cbar.mappable.get_clim())
    time_range = x_max - x_min

    if orientation == 'vertical':
        cbar.ax.yaxis_date()
        _configure_time_axis(cbar.ax.yaxis, time_range)
    elif orientation == 'horizontal':
        cbar.ax.xaxis_date()
        _configure_time_axis(cbar.ax.xaxis, time_range)
        plt.setp(cbar.ax.get_xticklabels(), rotation=45, ha='right')


def plot_with_colorbar(
        ax: Axes,
        data: pd.DataFrame,
        x: str,
        y: str,
        color_by: str,
        cmap: str,
        edgecolor: str,
        size: float | np.ndarray,
        alpha: float,
        legend: str = None,
        cbar_orientation: str = 'horizontal',
        cbar_pad: float = 0.06,
        cbar_aspect: int = 40,
        cbar_shrink: float = 0.6
) -> PathCollection:
    """
    Plots a scatter plot on the given axes with an associated colorbar.
    """
    if color_by not in data.columns:
        raise ValueError(f"Column '{color_by}' not found in data.")

    colorbar_label = color_by
    if color_by == 'mag':
        colorbar_label = 'Magnitude'
    elif color_by == 'time':
        colorbar_label = 'Origin time'
    elif color_by == 'depth':
        colorbar_label = 'Depth [km]'

    color = data[color_by]
    if color_by == 'time':
        color_numeric = mdates.date2num(color)
        global_min = mdates.date2num(color.min())
        global_max = mdates.date2num(color.max())
    else:
        color_numeric = color
        global_min = np.floor(color.min())
        global_max = np.ceil(color.max())

    scatter = ax.scatter(
        x=data[x], y=data[y], c=color_numeric, cmap=cmap,
        edgecolor=edgecolor, s=size, alpha=alpha, linewidth=0.25,
        vmin=global_min, vmax=global_max, label=legend
    )

    cbar = plt.colorbar(
        scatter, ax=ax, orientation=cbar_orientation,
        pad=cbar_pad, shrink=cbar_shrink, aspect=cbar_aspect
    )
    cbar.set_label(colorbar_label, fontsize=14)
    
    if color_by == 'time':
        format_colorbar_time(cbar, cbar_orientation)

    return scatter


def save_figure(
        save_name: str,
        save_extension: str = 'jpg',
        directory: str = './seismoviz_figures'
) -> None:
    """
    Saves the given figure to a file with the specified name, extension,
    and directory.
    """
    os.makedirs(directory, exist_ok=True)
    fig_name = os.path.join(directory, f'{save_name}.{save_extension}')
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {fig_name}")


def process_size_parameter(
    size: float | str,
    data: pd.DataFrame,
    size_scale_factor: tuple[float, float]
) -> float | np.ndarray:
    """
    Processes the size parameter and returns the computed size values for plotting.
    """
    if isinstance(size, (int, float)):
        return size
    elif isinstance(size, str):
        if size not in data.columns:
            raise ValueError(f"Column '{size}' not found in data.")
        return (data[size] * size_scale_factor[0]) ** size_scale_factor[1]
    else:
        raise ValueError(
            "The 'size' parameter must be a scalar or a column from your data."
        )


def create_size_legend(
        ax: Axes,
        size: str,
        data: pd.DataFrame,
        size_scale_factor: tuple[float, float],
        alpha: float,
        size_legend_loc: str
) -> None:
    """
    Creates a size legend on the given Axes.
    """
    min_size = np.floor(data[size].min())
    max_size = np.ceil(data[size].max())
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

    legend = ax.legend(
        handles=size_handles,
        loc=size_legend_loc,
        fancybox=False,
        edgecolor='black',
        ncol=len(size_values),
    )
    ax.add_artist(legend)


def plot_highlighted_events(
        ax: Axes,
        data: pd.DataFrame,
        hl_ms: float,
        hl_size: float,
        hl_marker: str,
        hl_color: str,
        hl_edgecolor: str,
        x: str,
        y: str
) -> None:
    """
    Plots highlighted events on the given Axes.
    """
    large_quakes = data[data['mag'] > hl_ms]
    ax.scatter(
        x=large_quakes[x], y=large_quakes[y], c=hl_color, s=hl_size,
        marker=hl_marker, edgecolor=hl_edgecolor, linewidth=0.75,
        label=f'Events M > {hl_ms}'
    )


def _configure_time_axis(axis, time_range):
    """
    Configures a time axis (x or y) based on the range of time.
    """
    if time_range.days > 3650:
        axis.set_major_locator(mdates.YearLocator())
        axis.set_major_formatter(mdates.DateFormatter('%Y'))
        axis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    elif time_range.days > 730:
        axis.set_major_locator(mdates.YearLocator())
        axis.set_major_formatter(mdates.DateFormatter('%Y'))
        axis.set_minor_locator(mdates.MonthLocator())
    elif time_range.days > 60:
        axis.set_major_locator(mdates.MonthLocator())
        axis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        axis.set_minor_locator(mdates.DayLocator(bymonthday=[15]))
    else:
        axis.set_major_locator(mdates.WeekdayLocator())
        axis.set_major_formatter(mdates.DateFormatter('%b %d'))
        axis.set_minor_locator(mdates.DayLocator())