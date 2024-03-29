Core Modules of MGT (Rigidity Graph)
======================================

:mod:`core` module consists of:
    1. The base :class:`core.BaseMG` that preprocess the fluctuogram data.
    2. The core :class:`core.CoreMG` that builds the rigidity graph matrix and decompose it.

To analyze inter-segments, use the ``interSegs`` argument with tuple of two segments. If the residue ids in two segments
overlap, the residue ids are renamed using :func:`_refactor_resid` for correct MG matrix creation.

.. automodule:: core
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance: