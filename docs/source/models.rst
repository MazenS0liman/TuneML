Models
======

This section summarizes the primary model families in TuneML.

Transformer Models
------------------

The transformer stack includes custom multi-head attention and feed-forward blocks,
as well as text-conditioned MIDI generation.

- Shared blocks: ``tuneml/models/transformer/Transformer.py``
- Text-conditioned generator: ``tuneml/models/transformer/MidiTransformer.py``
- GPT-style MIDI model: ``tuneml/models/transformer/MidiGPT.py``

For detailed transformer notes, see:
``../../tuneml/models/transformer/docs.md``

Audio Models
------------

- MiniVGG classifier: ``tuneml/models/vgg/MiniVGG.py``

Training Components
-------------------

- MIDI transformer trainer: ``tuneml/trainers/MidiTransformerTrainer.py``
- MIDI GPT trainer: ``tuneml/trainers/MidiGPTTrainer.py``
- Audio classifier trainer: ``tuneml/trainers/MiniVGGTrainer.py``
