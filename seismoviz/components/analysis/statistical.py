class StatisticalAnalyzer:
    def __init__(self, catalog: type):
        self.ct = catalog

    def calculate_interevent_time(self, unit='sec') -> None:
        """
        Calculates the inter-event time for sequential events in the catalog.

        .. note::
            After executing this method, a new column named ```interevent_time``` 
            will be created in the `catalog.data` DataFrame. This column will 
            contain the calculated inter-event times, making the data accessible 
            for further analysis or visualization.

        Parameters
        ----------
        unit : str, optional
            The time unit for the inter-event times. Supported values are:
            - ``'sec'``: seconds (default)
            - ``'min'``: minutes
            - ``'hour'``: hours
            - ``'day'``: days
        
        Raises
        ------
        ValueError
            If an unsupported time unit is provided.
        """
        valid_units = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400}
        if unit not in valid_units:
            raise ValueError(
                f"Unit '{unit}' not supported. Choose from {list(valid_units.keys())}."
            )
        
        self.ct.data.sort_values(by='time', ascending=True)
        
        time_diffs = self.ct.data['time'].diff().dt.total_seconds()
        time_diffs = time_diffs / valid_units[unit]

        self.ct.data['interevent_time'] = time_diffs

    def cov(self, window_size: int) -> None:
        """
        Calculates the coefficient of variation (COV) for the inter-event times 
        using a rolling window.

        .. note::
            After executing this method, a new column named ```cov``` will be 
            created in the `catalog.data` DataFrame. This column will contain 
            the calculated inter-event times, making the data accessible for 
            further analysis or visualization.

        Parameters
        ----------
        window_size : int
            The size of the rolling window (in number of events) over which the 
            coefficient of variation is calculated.
        """
        if 'interevent_time' not in self.ct.data.columns:
            self.calculate_interevent_time()

        rolling_mean = self.ct.data.interevent_time.rolling(
            window=window_size
        ).mean()
        rolling_std = self.ct.data.interevent_time.rolling(
            window=window_size
        ).std()

        rolling_cov = rolling_std / rolling_mean

        self.ct.data['cov'] = rolling_cov