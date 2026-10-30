<div align="center">
  <img src="static/images/tune-ml.png" alt="Tune ML logo" width="150"/>

  # Tune ML

  **A deep-learning toolkit for audio processing, audio classification, and music generation.**

  ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-FF8C00?logo=pytorch&logoColor=white)
</div>

---

## Overview

Tune ML combines reusable model implementations, training utilities, and experiment notebooks in a single repository. It currently includes:

- **Audio classification** with a compact VGG-style CNN (MiniVGG).
- **Text-conditioned MIDI generation** with an encoder-decoder Transformer.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Repository Layout](#repository-layout)
- [Models](#models)
  - [MiniVGG Audio Classifier](#minivgg-audio-classifier)
  - [MIDI Transformer](#midi-transformer)
- [Documentation](#documentation)

## Requirements

- Python 3.10 or newer
- A working PyTorch installation for your CPU or CUDA environment
- The dependencies declared in [`pyproject.toml`](pyproject.toml)

## Installation

Install the project with [Poetry](https://python-poetry.org/):

```bash
poetry install
```

To activate the virtual environment, run `poetry shell` (if available in your Poetry version), or use `poetry run <command>`.

Alternatively, install in editable mode with pip:

```bash
pip install -e .
```

## Repository Layout

| Path | Description |
| --- | --- |
| [`tuneml/models/`](tuneml/models/) | Model implementations, including the MiniVGG classifier and MIDI Transformers |
| [`tuneml/data/`](tuneml/data/) | Data-processing code |
| [`tuneml/tokenizers/`](tuneml/tokenizers/) | Tokenization code |
| [`tuneml/trainers/`](tuneml/trainers/) | Training utilities |
| [`tuneml/modules/`](tuneml/modules/) | Higher-level inference modules |
| [`notebooks/`](notebooks/) | Experiments for audio codecs, MIDI generation, text-to-MIDI, and audio classification |
| [`docs/source/`](docs/source/) | Sphinx documentation source |

## Models

### MiniVGG Audio Classifier

MiniVGG is a compact VGG-style convolutional neural network for audio classification. It expects a single-channel, 128 × 128 spectrogram.

#### Preprocessing and inference

The [`AudioClassifier`](tuneml/modules/classifier/AudioClassifier.py) inference module prepares audio as follows:

1. Load or receive a waveform and resample it to 22,050 Hz.
2. Convert multi-channel audio to mono.
3. Compute a 128-bin mel spectrogram (FFT size 2,048, hop length 512).
4. Convert the spectrogram to decibels and pad it to 128 × 128.
5. Pass it through three convolutional blocks, each with two convolutions, batch normalization, ReLU, and max pooling.
6. Classify the final features into the labels stored in the checkpoint.

The classifier returns either the most likely `(class_name, probability)` pair or the top `k` predictions.

> **Note:** A checkpoint should contain `model_state_dict` (or `state_dict`) and, preferably, a `class_to_idx` mapping.

#### Usage

```python
from tuneml.modules.classifier import AudioClassifier

classifier = AudioClassifier("weights/best_mini_vgg.pth", device="cpu")
prediction = classifier.predict("path/to/audio.wav")
print(prediction)
```

See the [VGG audio classifier notebook](notebooks/vgg_audio_classifier.ipynb) for training and evaluation experiments.

---

### MIDI Transformer

The MIDI Transformer is a text-conditioned encoder-decoder model that generates MIDI token sequences. It learns a distribution over the next MIDI token at each decoder position while conditioning on encoded text.

Listen to generated examples on the [samples page](https://mazens0liman.github.io/TuneML/samples.html).

#### Architecture

<div align="center">
  <img src="static/diagrams/midi-transformer.png" alt="MidiTransformer architecture" width="500"/>
</div>

1. Text tokens are embedded and encoded into a contextual memory tensor.
2. MIDI tokens are embedded and passed through masked self-attention.
3. The decoder cross-attends to the encoded text memory.
4. A final linear layer projects decoder states to MIDI vocabulary logits.

#### Components

| Component | Purpose |
| --- | --- |
| Text embedding | Converts text token IDs to `d_model` vectors |
| Text positional encoding | Injects absolute order information into the text sequence |
| Text encoder | Produces a contextual memory tensor from the text sequence |
| MIDI embedding | Converts MIDI token IDs to `d_model` vectors |
| MIDI positional encoding | Injects order information into the MIDI sequence |
| Transformer decoder | Applies masked self-attention over MIDI tokens and cross-attention over text memory |
| Final projection | Maps each decoder state to MIDI vocabulary logits |

#### API Reference

##### Building blocks: [`Transformer.py`](tuneml/models/transformer/Transformer.py)

Shared building blocks used by the decoder and other transformer-style models.

**`MultiHeadAttention`** projects queries, keys, and values into multiple heads, applies scaled dot-product attention in each head, and concatenates the result back to `d_model`. It is used for both masked self-attention and cross-attention.

| Input | Shape | Notes |
| --- | --- | --- |
| `q` | `(batch_size, seq_len_q, d_model)` | Queries |
| `k` | `(batch_size, seq_len_k, d_model)` | Keys |
| `v` | `(batch_size, seq_len_v, d_model)` | Values |
| `mask` | optional | Boolean mask; `True` entries are masked out |

Output: `(batch_size, seq_len_q, d_model)`

> **Note:** `d_model` must be divisible by `num_heads`.

**`PointwiseFFN`** applies a position-wise two-layer MLP to each token independently. It expands features to `d_ff`, applies `ReLU`, and projects back to `d_model`.

- Input: `x` with shape `(batch_size, seq_len, d_model)`
- Output: `(batch_size, seq_len, d_model)`

##### Model: [`MidiTransformer.py`](tuneml/models/transformer/MidiTransformer.py)

| Method | Description | Returns |
| --- | --- | --- |
| `tokenize_text(text, device="cpu")` | Tokenizes raw text with the Flan-T5 tokenizer | `text_tokens` `(batch_size, text_seq_len)` and `text_padding_mask` `(batch_size, text_seq_len)`, `True` at padded positions |
| `generate_causal_mask(seq_len, device=None)` | Builds an upper-triangular mask for autoregressive decoding | `(seq_len, seq_len)` boolean mask, `True` above the diagonal |
| `encode_text(text_tokens, text_padding_mask=None)` | Embeds text, adds positional encoding, applies dropout, and runs the encoder | `memory` `(batch_size, text_seq_len, d_model)` |
| `forward(text_tokens, midi_tokens, tgt_mask=None, text_padding_mask=None)` | Full conditional forward pass: encode text, embed MIDI, decode, project to logits | `(batch_size, midi_seq_len, midi_vocab_size)` |
| `generate(text_tokens, start_token_id, max_len=512, temperature=1.0, top_k=None, text_padding_mask=None, device="cpu")` | Autoregressively samples MIDI tokens | `(batch_size, generated_seq_len)` |

**How `generate` works**

1. Encode the text once.
2. Seed generation with the start token.
3. At each step, decode the current prefix.
4. Sample the next token from the last position's logits.
5. Append the token and repeat until `max_len`.

## Documentation

The project includes a Sphinx documentation site. Its sources are in [`docs/source/`](docs/source/).