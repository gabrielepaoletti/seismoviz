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

Modules overview
----------------
The core module provides foundational functionalities and utilities to initialize and work with seismic data.

.. grid:: 1 1 2 2
   :padding: 0 0 3 3
   :gutter: 2 2 3 3

   .. grid-item-card:: :octicon:`cpu;2.5em;sd-mr-1` SeismoViz Core
         :link: library/core
         :link-type: doc
         :link-alt: SeismoViz Core
      
         Essential functions for initializing and managing seismic catalogs, providing the foundation for further analysis and interaction.

Objects overview
----------------
SeismoViz revolves around two primary objects that enable seamless interaction with seismic data. Each object provides specialized methods and attributes designed to simplify workflows for analysis, visualization, and data manipulation.

.. grid:: 1 1 2 2
   :padding: 0 0 3 3
   :gutter: 2 2 3 3

   .. grid-item-card:: :octicon:`list-unordered;2.5em;sd-mr-1` Catalog
         :link: library/components/catalog
         :link-type: doc
         :link-alt: Catalog

         Contains methods for querying, filtering, and visualizing seismic data.

   .. grid-item-card:: :octicon:`graph;2.5em;sd-mr-1` CrossSection
         :link: library/components/cross_section
         :link-type: doc
         :link-alt: CrossSection 

         Tools to compute and visualize seismic cross-sections, providing insights into subsurface activity.