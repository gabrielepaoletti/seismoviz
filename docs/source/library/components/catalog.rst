.. title:: Catalog

.. image:: ../../_static/banners/catalog_light.jpg
   :align: center

--------------------

* `filter_events`
  Filters the catalog based on criteria such as time range, magnitude, or location.

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   filter_events <methods/common_filter>

Catalog
=======

.. autoclass:: seismoviz.components.catalog.Catalog
   :undoc-members:
   :member-order: bysource
   :exclude-members: inherited_method, inherited_property


Operation methods
------------------

.. automethod:: seismoviz.components.catalog.Catalog.filter
.. automethod:: seismoviz.components.catalog.Catalog.sort
.. automethod:: seismoviz.components.catalog.Catalog.deduplicate_events


Visualization methods
---------------------

.. automethod:: seismoviz.components.catalog.Catalog.plot_map
.. automethod:: seismoviz.components.catalog.Catalog.plot_space_time
.. automethod:: seismoviz.components.catalog.Catalog.plot_attribute_distributions


Magnitude analysis methods
--------------------------

.. automethod:: seismoviz.components.catalog.Catalog.plot_magnitude_time
.. automethod:: seismoviz.components.catalog.Catalog.fmd
.. automethod:: seismoviz.components.catalog.Catalog.estimate_b_value


Statistical analysis methods
----------------------------

.. automethod:: seismoviz.components.catalog.Catalog.plot_event_timeline
.. automethod:: seismoviz.components.catalog.Catalog.plot_interevent_time