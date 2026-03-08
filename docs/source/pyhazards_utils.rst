Utils
===================

Overview
--------

Utility helpers keep the rest of the library concise. Use these modules for
device selection, reproducibility, and small shared helpers.

Submodules
----------

- :mod:`pyhazards.utils.hardware`: device helpers and automatic device selection.
- :mod:`pyhazards.utils.common`: reproducibility, logging, and shared utility
  functions.

Typical Uses
------------

- choose CPU or GPU behavior explicitly,
- set deterministic seeds for experiments,
- reuse small helpers instead of copying project-specific boilerplate.
