import copy


class Operations:
    def __init__(self, instance: object):
        self._instance = instance
    
    def filter(self, **kwargs):
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
        """
        instance = copy.deepcopy(self._instance)

        for attribute, (criteria, value) in kwargs.items():
            if criteria == 'greater':
                instance.data = instance.data[instance.data[attribute] > value]
            elif criteria == 'lower':
                instance.data = instance.data[instance.data[attribute] < value]
            elif criteria == 'between':
                if isinstance(value, list) and len(value) == 2:
                    instance.data = instance.data[
                        instance.data[attribute].between(value[0], value[1])
                    ]
                else:
                    raise ValueError("Value must be a list of two numbers for 'between' criteria.")
            elif criteria == 'outside':
                if isinstance(value, list) and len(value) == 2:
                    instance.data = instance.data[
                        ~instance.data[attribute].between(value[0], value[1])
                    ]
                else:
                    raise ValueError("Value must be a list of two numbers for 'outside' criteria.")
            else:
                raise ValueError(
                    f"Invalid criteria '{criteria}'. Choose from 'greater', 'lower', "
                    "'between', or 'outside'."
                )

        return instance
    
    def sort(self, by: str, ascending: bool = True):
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
        """
        instance = copy.deepcopy(self._instance)
        instance.data = instance.data.sort_values(by=by, ascending=ascending)
        return instance
    
    def deduplicate_events(self, **kwargs):
        """
        Removes duplicate entries based on specific attributes.

        Returns
        -------
        object
            The same instance, with duplicate entries removed.
        """
        instance = copy.deepcopy(self._instance)
        if 'subset' not in kwargs:
            kwargs['subset'] = ['lon', 'lat', 'depth', 'time', 'mag']
    
        instance.data = instance.data.drop_duplicates(**kwargs)
        return instance