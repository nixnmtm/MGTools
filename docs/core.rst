Core Modules of MGT
======================================

:mod:`core` module consists of the base :class:`core.BuildMG` class that builds the molecular graph from the fluctuogram
data.

To analyze inter-segments, use the ``interSegs`` argument with tuple of two segments. If the residue ids in two segments
overlap, the residue ids are renamed using :func:`_refactor_resid` for correct MG matrix creation.

.. automodule:: core
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

