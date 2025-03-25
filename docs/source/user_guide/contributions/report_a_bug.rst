.. title:: Report a bug

.. image:: ../../_static/banners/issues_light.jpg
   :align: center

--------------------

Report a bug
============

Found a bug in SeismoViz? We appreciate your help in improving our software!

Before reporting
----------------

Before submitting a bug report, please:

1. **Check existing issues**: Search our `GitHub issues <https://github.com/gabrielepaoletti/seismoviz/issues>`_ to see if the bug has already been reported.

2. **Update your installation**: Ensure you're using the latest version of SeismoViz as the bug may have been fixed in a recent release.

3. **Verify it's a bug**: Make sure the behavior you're experiencing is actually a bug and not expected behavior.

How to submit a bug report
--------------------------

Please submit bug reports on our `GitHub Issues page <https://github.com/gabrielepaoletti/seismoviz/issues>`_ using the Bug Report template. Include:

1. **Descriptive title**: A clear, concise description of the issue.

2. **Environment details**:
   * SeismoViz version
   * Python version
   * Operating system
   * Installation method (pip, conda, from source, etc.)
   * Any other relevant environment details

3. **Steps to reproduce**:
   * Provide a minimal code example that reproduces the bug
   * Include sample data if relevant (or instructions to generate it)
   * List the exact steps to reproduce the behavior

4. **Expected behavior**: What you expected to happen.

5. **Actual behavior**: What actually happened. Include full error messages and tracebacks if applicable.

6. **Screenshots or plots**: If applicable, add screenshots to help explain your problem.

7. **Additional context**: Any other information that might be relevant.

Example bug report
------------------

.. code-block:: text

    Title: Plot function crashes when using logarithmic scale with negative values

    Environment:
    - SeismoViz version: 1.2.3
    - Python version: 3.9.5
    - OS: Ubuntu 20.04

    Steps to reproduce:
    ```python
    import seismoviz as smv
    import numpy as np
    
    data = np.linspace(-1, 1, 100)
    smv.plot(data, scale='log')
    ```

    Expected behavior:
    Warning about negative values and graceful handling or clear error message

    Actual behavior:
    Crashes with the following traceback:
    [paste traceback here]

    Additional context:
    This happens only when using the log scale with data containing negative values.

After submitting
----------------

* Watch for questions from maintainers who may need additional information.
* Be prepared to provide more details if requested.
* Consider submitting a pull request to fix the bug yourself!

Thank you for helping make SeismoViz better!