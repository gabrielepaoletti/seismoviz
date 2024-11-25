.. title:: API Reference

.. image:: _static/banners/api_light.jpg
   :align: center

--------------------

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Components
   
   library/core
   library/components/catalog
   library/components/cross_section


This reference manual provides an in-depth overview of the SeismoViz library, detailing its modules, functions, and core components. It is designed to assist users in managing seismic data effectively and implementing advanced analytical workflows.


.. raw:: html

    <h2 style="font-size: 1.2em;">Guidelines for importing SeismoViz</h2>

The recommended way to import SeismoViz is by using the shorthand alias ``sv``:

.. code-block:: python

   import seismoviz as sv

This approach ensures consistency across projects and keeps the code concise and readable.


.. raw:: html

    <h2 style="font-size: 1.2em;">Accessing library functions and classes</h2>

SeismoViz is designed around high-level methods and classes that streamline workflows for seismic data processing and analysis. The main functionalities are accessed as methods or attributes of specific objects.

.. raw:: html

    <h3 style="font-size: 1.2em;">Example workflow</h3>

Start by loading a seismic catalog using the ``sv.read_catalog`` function, which returns a ``Catalog`` object:

.. code-block:: python

   catalog = sv.read_catalog(path='path/to/catalog.csv')

Once you have the ``Catalog`` object, all subsequent operations can be performed using its methods and attributes:

.. code-block:: python

   catalog.method_name()

This design emphasizes object-oriented workflows, enabling intuitive chaining of operations while maintaining a clean namespace.

.. note::
   While Python allows access to private modules and methods, such usage is discouraged. Breaking changes may occur in future releases if private components are used.


.. raw:: html

    <h2 style="font-size: 1.2em;">API Definition</h2>

.. card:: Core
   :link: library/core
   :link-type: doc

   Contains essential functions for high-level operations.

.. card:: Catalog
   :link: library/components/catalog
   :link-type: doc

   Contains methods for visualization and analysis of seismic catalogs.

.. card:: Cross Sections
   :link: library/components/cross_section
   :link-type: doc

   Provides tools to compute and visualize seismic cross-sections.