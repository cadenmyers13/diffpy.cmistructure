=============
Release notes
=============

.. current developments

0.1.1
=====

**Added:**

* A "Migrating from diffpy.srfit.structure" documentation page that lists every renamed function, method, attribute and argument.
* The ``npintensity.py`` and ``npintensityII.py`` examples from diffpy.srfit, updated to the diffpy.cmistructure names.
* A "Making a structure adapter" documentation page that explains how to adapt another structure representation for use with diffpy.cmistructure.

**Changed:**

* Use the same license as ``diffpy.srfit``: ``LICENSE.rst``, the documentation License page and the ``License :: Free To Use But Restricted`` classifier now match it.
* Update Simon Billinge's contact email in ``LICENSE_DANSE.rst``.

**Fixed:**

* Require ``diffpy.srfit`` 3.4.0 or later. Older versions lack the methods diffpy.cmistructure calls, so installing it next to ``diffpy.srfit`` 3.3.1 failed at run time.

**Removed:**

* Support for Python 3.11. ``diffpy.srfit`` 3.4.0, which diffpy.cmistructure requires, needs Python 3.12 or later.


0.1.0
=====

**Added:**

* ``constrain_as_space_group`` also accepts a generator, such as a ``PDFGenerator``, whose ``.phase`` is the structure parameter set, for example ``constrain_as_space_group(generator, "Fm-3m")``.

