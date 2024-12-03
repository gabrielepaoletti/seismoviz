.. title:: CorssSection

.. image:: ../../_static/banners/cs_light.jpg
   :align: center

--------------------

Cross Sections
==============

.. autoclass:: seismoviz.components.cross_section.CrossSection


Operation methods
------------------

.. automethod:: seismoviz.components.cross_section.CrossSection.filter
.. automethod:: seismoviz.components.cross_section.CrossSection.sort
.. automethod:: seismoviz.components.cross_section.CrossSection.deduplicate_events


Visualization methods
---------------------

.. automethod:: seismoviz.components.cross_section.CrossSection.plot_sections
.. automethod:: seismoviz.components.cross_section.CrossSection.plot_section_lines


Magnitude analysis methods
--------------------------

.. automethod:: seismoviz.components.cross_section.CrossSection.magnitude_time
.. automethod:: seismoviz.components.cross_section.CrossSection.fmd
.. automethod:: seismoviz.components.cross_section.CrossSection.estimate_b_value


Statistical analysis methods
----------------------------

.. automethod:: seismoviz.components.cross_section.CrossSection.event_timeline
.. automethod:: seismoviz.components.cross_section.CrossSection.interevent_time
.. automethod:: seismoviz.components.cross_section.CrossSection.cov