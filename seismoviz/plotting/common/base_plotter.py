import os
import matplotlib.pyplot as plt


class BasePlotter:
    def __init__(self, style: dict[str, str] = None) -> None:
        """
        Initializes the BasePlotter with a specific style configuration.

        Parameters
        ----------
        style : dict of str, optional
            A dictionary defining style attributes for the plot (e.g., 
            font size, line width, colors).
        """
        self.style = style

    def save_figure(
        self, 
        save_name: str, 
        save_extension: str = 'jpg', 
        directory: str = './seismoviz_figures'
    ) -> None:
        """
        Saves the given figure to a file with the specified name, extension, 
        and directory.

        Parameters
        ----------
        save_name : str
            The base name used for saving the figure. It serves as the prefix 
            for the file name.

        save_extension : str, optional
            The file extension to use when saving figures (e.g., 'jpg', 'png'). 
            The default extension is 'jpg'.

        directory : str, optional
            The directory where the figure will be saved. Defaults to 
            './seismoviz_figures'.

        Returns
        -------
        None
        """
        os.makedirs(directory, exist_ok=True)
        fig_name = os.path.join(directory, f'{save_name}.{save_extension}')
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {fig_name}")
