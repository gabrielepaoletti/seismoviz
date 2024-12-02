import numpy as np
import pandas as pd


class GeospatialMixin:
    def __init__(self) -> None:
        self.zone = self.utm_zone()
        self.hemisphere = self.hemisphere()

    def utm_zone(self) -> int:
        """
        Determines the most common UTM (Universal Transverse Mercator) zone 
        based on the longitude values of the seismic events in the dataset.
        """
        utm_zones = np.int_((np.array(self.data.lon) + 180) / 6) + 1
        zones, counts = np.unique(utm_zones, return_counts=True)
        utm_zone = zones[np.argmax(counts)]
        return utm_zone.item()

    def hemisphere(self) -> str:
        """
        Determines the predominant hemisphere (either North or South) based on 
        the latitude values of the seismic events in the dataset.
        """
        norths = np.sum(np.array(self.data.lat) >= 0)
        souths = len(self.data.lat) - norths
        return 'north' if norths >= souths else 'south'


class DunderMethodMixin:
    def __init__(self) -> None:
        pass

    def __eq__(self, other) -> bool:
        """
        Checks if two instances are equal based on their data.
        """
        return self.data.equals(other.data)

    def __getitem__(self, key: tuple[int, int] | int) -> pd.Series | pd.DataFrame:
        """
        Returns the row as a pandas Series or the sub-DataFrame at the 
        specified index.

        If a tuple (section_id, row_index) is provided, returns the row as 
        a pandas Series.

        If only an integer is provided and the DataFrame is MultiIndexed, 
        returns the sub-DataFrame for that section_id.
        """
        if isinstance(self.data.index, pd.MultiIndex):
            if isinstance(key, tuple):
                section_id, row_index = key
                section_df = self.data.xs(section_id, level='section_id')
                return section_df.iloc[row_index]
            else:
                return self.data.xs(key, level='section_id')
        else:
            return self.data.iloc[key]

    def __len__(self) -> int | tuple[int, int]:
        """
        Returns the number of seismic events.
        """
        if isinstance(self.data.index, pd.MultiIndex):
            return [(level, len(self.data.loc[level])) for level in self.data.index.levels[0]]
        else:
            return len(self.data)

    def __repr__(self) -> str:
        """
        Generate a string representation of the instance.
        """
        parts = [f"{self.__class__.__name__}("]
        for key, value in self.__dict__.items():
            if key.startswith('_') or key in ['data', 'instance']:
                continue
            elif isinstance(value, str) and value.startswith("<") and value.endswith(">"):
                parts.append(f"    {key}={value}")
            else:
                parts.append(f"    {key}={repr(value)}")
        parts.append(")")
        return "\n".join(parts)
