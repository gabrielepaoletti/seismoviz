.. title:: Contribute

--------------------

Contribute
==========

Thank you for your interest in contributing to SeismoViz! This guide will help you get started with contributing code, following our coding standards, and understanding our development practices.


How to contribute
-----------------

1. **Fork the repository**
    - Visit the `SeismoViz GitHub repository <https://github.com/gabrielepaoletti/seismoviz>`_.
    - Click the Fork button in the top-right corner to create a copy of the repository under your GitHub account.

2. **Clone your fork**
    - Clone the forked repository to your local machine:
      
      .. code-block:: bash

        git clone https://github.com/gabrielepaoletti/seismoviz

3. **Create a branch**
    - Navigate to the repositort directory:
      
      .. code-block:: bash

        cd seismoviz
    
    - Create a new branch for your feature or bug fix:
      
      .. code-block:: bash

        git checkout -b feature/your-feature-name

4. **Make your changes**
    - Implement your code changes, ensuring you follow the `Code Style Guidelines <https://seismoviz.readthedocs.io/en/latest/user_guide/contributions/contribute.html#code-style-guidelines>`_.
    - When adding methods to main classes, only include wrapper methods decorated with ``@sync_metadata``. Internal implementation should be handled in separate modules or components.

5. **Write tests**
    - Add unit tests to cover your new code.
    - Ensure that all existing tests pass by running:

      .. code-block:: bash

        pytest

6. **Commit your changes**
    - Stage your changes:

      .. code-block:: bash

        git add .

    - Commit with a descriptive message:

      .. code-block:: bash

        git commit -m "Add feature: your feature description"

7. **Push to GitHub**
    - Push your branch to your forked repository:

      .. code-block:: bash

        git push origin feature/your-feature-name

8. **Open a Pull Request**
    - Go to your fork on GitHub.
    - Click the Compare & pull request button.
    - Provide a clear description of your changes and any relevant information.


Code style guidelines
---------------------

To maintain code consistency and readability, please adhere to the following guidelines.


Follow PEP 8 Standards
^^^^^^^^^^^^^^^^^^^^^^

`PEP 8 is the official Python style guide <https://peps.python.org/pep-0008/>`_ that outlines how to format Python code for maximum readability.


Key points
~~~~~~~~~~
- **Indentation:** Use 4 spaces per indentation level.
- **Line length:** Limit all lines to a maximum of 79 characters.
- **Blank lines:** Use blank lines to separate functions and classes.
- **Imports:** Place all imports at the top of the file.
- **Naming conventions:** Use lowercase with underscores for functions and variables; use ``CapWords`` for classes.

Make sure to read the official document by Python, to fully understand the guidelines.


Write docstrings in NumPy Docstring format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ provides a standard for writing well-structured and readable docstrings. Following these guidelines is critical to ensure that docstrings are processed correctly by documentation providers, such as Sphinx or other automatic generation tools.

.. code-block:: python

    def read_catalog(path: str, **kwargs) -> Catalog:
        """
        Reads a CSV file and returns a ``Catalog`` object.

        Parameters
        ----------
        path : str
            The path to the CSV file containing the seismic catalog.

        **kwargs
            Additional keyword arguments to pass to ``pandas.read_csv()``.
        
        Returns
        -------
        Catalog
            An instance of the ``Catalog`` class with the data loaded.

        Raises
        ------
        ValueError
            If the CSV file does not contain defined columns.

        Examples
        --------

        Basic usage:
        
        .. code-block:: python

            # Reading a catalog with default settings
            catalog = sv.read_catalog(
                path='seismic_data.csv'
            )

        For a more customized behavior, you can pass ``pd.read_csv()`` arguments:

        .. code-block:: python

            # Reading a catalog with a custom delimiter and selected columns
            catalog = sv.read_catalog(
                path='seismic_data.csv', 
                delimiter=';', 
                usecols=['id', 'lon', 'lat', 'depth', 'time', 'mag']
            )
        
        .. warning::
            The input CSV file must contain the following columns: 
            ``lon``, ``lat``, ``time``, ``depth``, ``mag``, and ``id``.
            If any of these columns are missing, an error will be raised.
        """
        data = pd.read_csv(path, parse_dates=['time'], **kwargs)
        return Catalog(data)


Use decorators appropriately
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When adding methods to main classes, use the ``@sync_metadata`` decorator to ensure metadata consistency between the main class and its components.

In the main classes, include only wrapper methods that interface with internal components. These methods should delegate functionality to specialized modules or classes, keeping the main class interface clean and focused.

.. code-block:: python

        class ComponentClass:
            """This class contains all the implementations"""
            def method(self):
                # Method implementation
                return self
        
        class MainClass:
            """This class contains just the wrappers for readability"""
            @sync_metadata(ComponentClass, 'method')
            def method(self, **kwargs):
                return ComponentClass.method(**kwargs)

Do not include complex logic or implementation details in the main classes. Keep the main classes focused on providing a user-friendly interface.

--------------------

Thank you for contributing to SeismoViz! Your support helps us improve and expand this project. If you have any questions or need assistance, feel free to reach out through the GitHub repository.