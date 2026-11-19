"""Naksha AI model 3.0 - standalone LiDAR classification.

The package is independent of the Naksha LiDAR application: it imports nothing from Naksha's
``gui`` package. Naksha launches it as a separate program; it can also be used on its own
through its command line (``python -m naksha_ai3``) or, from phase 2, its own window.

Layout:
    engine/      the classification engine (PDAL + Myria3D + rule engine), run as a subprocess
                 in the AI3 runtime environment
    runtime.py   locates and verifies that runtime environment
    controller.py  starts, monitors and cancels one engine run
    ptc.py       reads a .ptc class-definition file and resolves its semantic classes
    mapping.py   class codes the engine and the post-processing use
    lasio.py     LAS/LAZ reading, cropping and classified-copy writing
    fences.py    fence polygons -> point indices
    reference.py learning from already classified areas inside fences
    job.py       one classification job (full file or fences) and batches of them
    cli.py       the ``naksha-ai3`` command line
"""

__version__ = "3.0.1"
