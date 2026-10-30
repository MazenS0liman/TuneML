Project Overview
================

TuneML is a Python project for audio processing, classification, and
generation with deep learning. The repository combines reusable model
implementations, training utilities, datasets, and notebooks for
experimentation.

Features
--------

- **Audio classification** with MiniVGG, a compact VGG-style CNN.
- **Text-conditioned MIDI generation** with an encoder-decoder Transformer.
- **Training utilities** for each model family.
- **Notebooks** for MIDI generation, and audio classification experiments.

Requirements
------------

- Python 3.10 or newer
- A working PyTorch installation for your CPU or CUDA environment
- The dependencies declared in ``pyproject.toml``

Installation
------------

Install the project with `Poetry <https://python-poetry.org/>`_:

.. code-block:: bash

   poetry install

To activate the virtual environment, run ``poetry shell`` (if available in your
Poetry version) or prefix commands with ``poetry run``. Alternatively, install
in editable mode with pip:

.. code-block:: bash

   pip install -e .

Quick Start
-----------

Classify an audio file with a trained MiniVGG checkpoint:

.. code-block:: python

   from tuneml.modules.classifier import AudioClassifier

   classifier = AudioClassifier("weights/best_mini_vgg.pth", device="cpu")
   prediction = classifier.predict("path/to/audio.wav")
   print(prediction)

See :doc:`models` for details on each model and :doc:`api` for the full
reference.

Repository Layout
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Description
   * - ``tuneml/models/``
     - Model architectures, including MiniVGG and the MIDI Transformers
   * - ``tuneml/data/``
     - Data-processing code
   * - ``tuneml/tokenizers/``
     - Tokenization code
   * - ``tuneml/trainers/``
     - Training loops and trainers
   * - ``tuneml/modules/``
     - Higher-level inference modules
   * - ``notebooks/``
     - Experiments for audio codecs, MIDI generation, text-to-MIDI, and audio
       classification
   * - ``docs/source/``
     - Sphinx documentation source
