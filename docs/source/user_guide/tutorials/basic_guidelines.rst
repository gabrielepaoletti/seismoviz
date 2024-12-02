.. title:: Basic usage guidelines

--------------------

Basic usage guidelines
======================

This guide outlines the fundamental principles and best practices for using SeismoViz effectively. Learn how to import the library, interact with its core objects, and access documentation inline to simplify your workflows.

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

Accessing documentation inline
------------------------------

SeismoViz provides comprehensive docstrings for its methods and classes, allowing you to access detailed documentation directly within your Python environment. This is especially useful for understanding the functionality and parameters of specific methods or objects without leaving your development workflow.

To view the docstring of any method or class, simply append a ``?`` to its name in an interactive Python session or Jupyter Notebook:

.. code-block:: python

   instance.method_name?

.. tip::
   Using ``help(object)`` is another way to access the docstring. However, the ``?`` syntax is often more concise and integrates seamlessly with interactive environments.
