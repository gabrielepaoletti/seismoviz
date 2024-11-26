class Operations:
    @staticmethod
    def filter(instance, **kwargs):
        """
        Filters the instance's dataset based on multiple specified conditions.

        .. note::
            Each keyword argument should be in the form 
            ``attribute=('criteria', value)`` or 
            ``attribute=('criteria', [value1, value2])`` for range criteria. 
            This allows for intuitive and flexible filtering based on the 
            attributes of the data.

        Parameters
        ----------
        instance : object
            The instance containing the dataset (must have an attribute `data`).

        **kwargs : dict
            Arbitrary keyword arguments representing the filtering 
            conditions. Each key is an attribute name in the dataset, 
            and the value is a tuple specifying the criteria (``'greater'``, 
            ``'lower'``, ``'between'``, ``'outside'``) and the comparison value(s).

        Returns
        -------
        object
            The same instance, with its dataset filtered.

        Examples
        --------
        To filter the dataset for entries with magnitude greater than 4.5, depth 
        between the range 10-50 km, and time earlier than October 30, 2016:

        .. code-block:: python

            filtered_instance = instance.filter(
                mag=('greater', 4.5),
                depth=('between', [10, 50]),
                time=('lower', '2016-10-30')
            )
        """
        filtered_data = instance.data

        for attribute, (criteria, value) in kwargs.items():
            if criteria == 'greater':
                filtered_data = filtered_data[filtered_data[attribute] > value]

            elif criteria == 'lower':
                filtered_data = filtered_data[filtered_data[attribute] < value]

            elif criteria == 'between':
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(
                        "Value must be a list of two numbers for 'between' criteria."
                    )
                filtered_data = filtered_data[
                    filtered_data[attribute].between(value[0], value[1])
                ]

            elif criteria == 'outside':
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(
                        "Value must be a list of two numbers for 'outside' criteria."
                    )
                filtered_data = filtered_data[
                    ~filtered_data[attribute].between(value[0], value[1])
                ]

            else:
                raise ValueError(
                    f"Invalid criteria '{criteria}'. Choose from 'greater', 'lower', "
                    "'between', or 'outside'."
                )

        instance.data = filtered_data
        return instance

    @staticmethod
    def sort(instance, by: str, ascending: bool = True):
        """
        Sorts the instance's dataset by a specific attribute.

        Parameters
        ----------
        instance : object
            The instance containing the dataset (must have an attribute `data`).

        by : str
            The attribute in the dataset to sort by.

        ascending : bool, optional
            Determines the sorting order. If ``True`` (default), sorts in 
            ascending order; if ``False``, in descending order.

        Returns
        -------
        object
            The same instance, with its dataset sorted.

        Examples
        --------
        To sort the dataset by time in ascending order:

        .. code-block:: python

            sorted_instance = instance.sort(
                by='time',
                ascending=True
            )
        """
        instance.data = instance.data.sort_values(by=by, ascending=ascending)
        return instance

    @staticmethod
    def deduplicate_events(instance):
        """
        Removes duplicate entries based on specific attributes.

        Parameters
        ----------
        instance : object
            The instance containing the dataset (must have an attribute `data`).

        Returns
        -------
        object
            The same instance, with duplicate entries removed.

        Examples
        --------
        To remove duplicates from the dataset:

        .. code-block:: python

            deduplicated_instance = instance.deduplicate_events()

        Notes
        -----
        This method considers duplicates based on the combination of the 
        following attributes: ``'lon'``, ``'lat'``, ``'depth'``, and ``'time'``.
        """
        instance.data = instance.data.drop_duplicates(
            subset=['lon', 'lat', 'depth', 'time']
        )
        return instance