Wildfire Interactive Map
========================

PyHazards includes a lightweight launcher for the external **RAI Fire**
interactive wildfire map. This companion site is specific to wildfire use
cases; it is not a general interactive map for every hazard domain in
PyHazards.

Use it when you want a browser-based wildfire view without leaving the broader
PyHazards workflow.

What This Page Covers
---------------------

- the live wildfire-focused RAI Fire website,
- the built-in launcher command,
- the small Python helper exposed by the package.

Live Website
------------

- `RAI Fire <https://rai-fire.com/>`_
- `Source repository <https://github.com/LabRAI/wildfire_interactive_map>`_

Command Line
------------

Open the website from the library with:

.. code-block:: bash

    python -m pyhazards map

The command prints the URL and, when possible, opens it in your default browser.

Python API
----------

.. code-block:: python

    from pyhazards import open_interactive_map

    url = open_interactive_map()
    print(url)

Notes
-----

The interactive map is an external companion application for wildfire
visualization. PyHazards links to it and provides a launcher, but it does not
host the web application inside the Python package itself.

Module Reference
----------------

.. automodule:: pyhazards.interactive_map
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
