Interactive Map
===============

PyHazards links to the external **RAI Fire** interactive wildfire map for browser-based
exploration and public sharing.

Live Website
------------

- `RAI Fire <https://rai-fire.com/>`_
- `Source repository <https://github.com/LabRAI/wildfire_interactive_map>`_

Command Line
------------

Use the built-in launcher to open the website from the library:

.. code-block:: bash

    python -m pyhazards map

The command always prints the website URL and attempts to open it in your default browser.

Python API
----------

.. code-block:: python

    from pyhazards import open_interactive_map

    url = open_interactive_map()
    print(url)

Module Reference
----------------

.. automodule:: pyhazards.interactive_map
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
