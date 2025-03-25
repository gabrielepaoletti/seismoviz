.. title:: Contribute

.. image:: ../../_static/banners/contribute_light.jpg
   :align: center

--------------------

Contribute
==========

Thank you for your interest in contributing to SeismoViz! This guide will help you get started with contributing code, following our coding standards, and understanding our development practices.

Setting up your development environment
---------------------------------------

1. **Fork the repository**: Visit the `SeismoViz GitHub repository <https://github.com/gabrielepaoletti/seismoviz>`_ and click the "Fork" button to create your own copy.

2. **Clone your fork**:

   .. code-block:: bash

      git clone https://github.com/gabrielepaoletti/seismoviz.git
      cd seismoviz

3. **Set up a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows, use: venv\Scripts\activate

4. **Install development dependencies**:

   .. code-block:: bash

      pip install -e ".[dev]"

Development workflow
--------------------

1. **Create a new branch** for your feature or bugfix:

   .. code-block:: bash

      git checkout -b feature-name

2. **Make your changes**: Write your code following our coding standards (see below).

3. **Add tests**: Ensure that your code is properly tested.

4. **Update documentation**: Add or update documentation as needed.

5. **Run tests locally**:

   .. code-block:: bash

      pytest

6. **Commit your changes**:

   .. code-block:: bash

      git add .
      git commit -m "Your descriptive commit message"

7. **Push your branch to GitHub**:

   .. code-block:: bash

      git push origin feature-name

8. **Open a Pull Request**: Go to the `SeismoViz repository <https://github.com/gabrielepaoletti/seismoviz>`_ and open a pull request from your branch.

Coding standards
----------------

We follow these standards for our code:

* **Code style**: We use PEP 8 for code style. Please use `black <https://black.readthedocs.io/>`_ for automatic formatting:

  .. code-block:: bash

     black .

* **Type hints**: Use type hints whenever possible.

* **Docstrings**: Write docstrings in NumPy style for all public functions, classes, and methods.

  .. code-block:: python

     def function(param1, param2):
         """
         Short description of the function.

         Parameters
         ----------
         param1 : type
             Description of param1
         param2 : type
             Description of param2

         Returns
         -------
         type
             Description of return value
         """
         return result

* **Import order**: Group imports in the following order, with a blank line between each group:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports

Testing
-------

* Write unit tests for all new functionality.
* Ensure that your tests cover both normal use cases and edge cases.
* Make sure all tests pass before submitting a pull request.

Documentation
------------

* Update documentation for any new features or changes to existing functionality.
* Follow the RST formatting used throughout our documentation.
* Include examples where appropriate.

Pull request process
--------------------

1. Ensure all tests pass.
2. Update the README.md and documentation if needed.
3. The PR should be associated with an issue when appropriate.
4. Your PR will be reviewed by maintainers who may request changes.
5. Once approved, a maintainer will merge your contribution.

Questions?
----------

If you have any questions or need help with your contribution, feel free to:

* Open an issue on GitHub
* Reach out to the maintainers
* Join our community channels

Thank you for contributing to SeismoViz!