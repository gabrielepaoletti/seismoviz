import numpy as np
import pandas as pd


class GeospatialMixin:
    def __init__(self) -> None:
        self.zone = self.utm_zone()
        self.hemisphere = self.hemisphere()

    def utm_zone(self) -> int:
        """
        Determines the most common UTM (Universal Transverse Mercator) zone based on the longitude values of the seismic events in the dataset.

        Returns
        -------
        int
            The most common UTM zone number among the seismic events' longitude values. The UTM zone is determined as an integer representing one of the 60 longitudinal projection zones used in the UTM system.
        """
        utm_zones = np.int_((np.array(self.data.lon) + 180) / 6) + 1
        zones, counts = np.unique(utm_zones, return_counts=True)
        utm_zone = zones[np.argmax(counts)]
        return utm_zone

    def hemisphere(self) -> str:
        """
        Determines the predominant hemisphere (either North or South) based on the latitude values of the seismic events in the dataset.
        
        Returns
        -------
        str
            A string indicating the predominant hemisphere where the seismic events are located. Returns 'north' if the majority of events are in the Northern Hemisphere and 'south' if the majority are in the Southern Hemisphere.
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

        Parameters
        ----------
        other : Any
            Another instance to compare with.

        Returns
        -------
        bool
            True if the two catalogs contain identical data, False otherwise.
        """
        return self.data.equals(other.data)
    
    def __getitem__(self, key: tuple | int) -> pd.Series | pd.DataFrame:
        """
        Returns the row as a pandas Series or the sub-DataFrame at the specified index.
        
        If a tuple (section_id, row_index) is provided, returns the row as a pandas Series.
        
        If only an integer is provided and the DataFrame is MultiIndexed, returns the sub-DataFrame for that section_id.

        Parameters
        ----------
        key : tuple or int
            If a tuple, the first element is 'section_id' and the second is 'row_index'.
            If an int, and the DataFrame is MultiIndexed, it returns the sub-DataFrame 
            for that section_id. If the DataFrame is not MultiIndexed, it returns the row at this index as a Series.

        Returns
        -------
        pd.Series or pd.DataFrame
            A Series or DataFrame containing the requested row or sub-DataFrame.
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

        Returns
        -------
        Union[int, tuple[int, int]]
            The number of seismic events if the index is not MultiIndex,
            otherwise, a tuple containing the index and the number of seismic events for that index.
        """
        if isinstance(self.data.index, pd.MultiIndex):
            return [(level, len(self.data.loc[level])) for level in self.data.index.levels[0]]
        else:
            return len(self.data)
    
    def __repr__(self) -> str:
        """
        Generate a string representation of the instance.

        Returns
        -------
        str
            A string representation of the instance, formatted with the class name and key-value pairs of the relevant attributes.
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