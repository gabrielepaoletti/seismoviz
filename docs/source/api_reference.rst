.. title:: API Reference

.. image:: _static/banners/api_light.jpg
   :align: center

--------------------

.. toctree::
   :maxdepth: 0
   :hidden:
   :caption: Components
   
   library/core
   library/components/catalog
   library/components/cross_section

API Reference
=============

This reference manual provides an in-depth overview of the SeismoViz library, detailing its modules, functions, and core components. It is designed to assist users in managing seismic data effectively and implementing advanced analytical workflows.


Guidelines for importing
------------------------

The recommended way to import SeismoViz is by using the shorthand alias ``sv``:

.. code-block:: python

   import seismoviz as sv

This approach ensures consistency across projects and keeps the code concise and readable.


Accessing library functions and classes
---------------------------------------

SeismoViz is designed around high-level methods and classes that streamline workflows for seismic data processing and analysis. The main functionalities are accessed as methods or attributes of specific objects.

Example workflow
^^^^^^^^^^^^^^^^

Start by loading a seismic catalog using the ``sv.read_catalog`` function, which returns a ``Catalog`` object:

.. code-block:: python

   catalog = sv.read_catalog(path='path/to/catalog.csv')

Once you have the ``Catalog`` object, all subsequent operations can be performed using its methods and attributes:

.. code-block:: python

   catalog.method_name()

This design emphasizes object-oriented workflows, enabling intuitive chaining of operations while maintaining a clean namespace.

.. note::
   While Python allows access to private modules and methods, such usage is discouraged. Breaking changes may occur in future releases if private components are used.


API Definition
--------------
To begin using SeismoViz, users should start with the core module, which provides essential functions and utilities to initialize workflows.


.. grid:: 1 1 3 3
   :padding: 0 0 3 3
   :gutter: 2 2 3 3

   .. grid-item-card:: :octicon:`apps;2.5em;sd-mr-1` SeismoViz Core
         :link: library/core
         :link-type: doc
         :link-alt: SeismoViz Core

         Contains essential functions for high-level operations.


Key objects
^^^^^^^^^^^
SeismoViz revolves around three primary objects that streamline your interaction with seismic data:

.. grid:: 1 1 3 3
   :padding: 0 0 3 3
   :gutter: 2 2 3 3

   .. grid-item-card:: :octicon:`project-roadmap;2.5em;sd-mr-1` Catalog
         :link: library/components/catalog
         :link-type: doc
         :link-alt: Catalog

         Contains methods for visualization and analysis of seismic catalogs.

   .. grid-item-card:: :octicon:`project;2.5em;sd-mr-1` CrossSection
         :link: library/components/cross_section
         :link-type: doc
         :link-alt: CrossSection

         Provides tools to compute and visualize seismic cross-sections.

   .. grid-item-card:: :octicon:`project-roadmap;2.5em;sd-mr-1` SubCatalog
         :link: library/components/sub_catalog
         :link-type: doc
         :link-alt: SubCatalog

         Contains methods for visualization and analysis of catalog subsets.