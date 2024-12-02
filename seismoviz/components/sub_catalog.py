import pandas as pd

from seismoviz.components import Catalog
from seismoviz.internal.decorators import sync_metadata
from seismoviz.components.visualization import SubCatalogPlotter

class SubCatalog(Catalog):
    def __init__(self, data: pd.DataFrame, selected_from: str) -> None:
        super().__init__(data)
        
        self.selected_from = selected_from
        self._sc_plotter = SubCatalogPlotter(self)

    @sync_metadata(SubCatalogPlotter, 'plot_on_section')
    def plot_on_section(self, **kwargs):
        self._sc_plotter.plot_on_section(**kwargs)