# The Whisper Model

Background notes on the model whispr is built around. For usage, see the
[README](README.md); this document covers the architecture in more depth.

[Whisper](https://openai.com/research/whisper) is an automatic speech recognition
(ASR) model published by OpenAI in September 2022
([Radford et al., arXiv:2212.04356](https://arxiv.org/abs/2212.04356)). It is a
sequence-to-sequence model based on a standard **encoder-decoder Transformer**
architecture, trained end-to-end with weak supervision on a large-scale
multilingual dataset.

## Audio preprocessing

Raw audio undergoes the following transformations before reaching the model:

1. Resampled to **16 kHz mono**.
2. A **log-Mel spectrogram** is computed with a 25 ms window, 10 ms stride, and **80 Mel frequency bins**.
3. The spectrogram is split into **30-second windows** of 3000 frames (padded with silence if shorter).

The resulting 80 x 3000 tensor is the input to the encoder.

## Encoder

The encoder consists of a convolutional front-end followed by Transformer blocks:

- Two 1D convolution layers (kernel size 3, GELU activation) downsample the time axis by a factor of 2, reducing the sequence from 3000 to 1500 frames.
- A stack of Transformer encoder blocks with multi-head self-attention and learned sinusoidal positional embeddings produces a sequence of contextual hidden representations.

Hyperparameters scale with model size (`base`: 6 layers, 512 dims, 8 heads; `large-v2`: 32 layers, 1280 dims, 20 heads).

## Decoder

The decoder generates text tokens autoregressively via cross-attention over the encoder output:

- A multilingual **Byte-Pair Encoding (BPE)** tokenizer with a vocabulary of ~50,000 tokens.
- Each generation is conditioned on a structured prompt of special tokens:
  - `<|startoftranscript|>`: sequence boundary
  - `<|fr|>`, `<|en|>`, ...: target language (omitted for auto-detect)
  - `<|transcribe|>` or `<|translate|>`: task selector
  - `<|notimestamps|>` or `<|0.00|>`: disables or enables timestamp prediction

## Timestamp prediction

When timestamp mode is active, the decoder interleaves **timestamp tokens**
(`<|0.00|>` to `<|30.00|>`, 0.02 s resolution) with text tokens, enabling
segment-level timing without any external alignment step.

## Language detection

When no language is specified, Whisper runs the encoder on the first 30 seconds,
then takes a softmax over language tokens at the first decoder step to identify
the language. This adds ~1 to 2 seconds of overhead and works reliably for the 99
languages present in the training data.

## Training

Whisper was trained on **680,000 hours** of multilingual audio paired with
transcripts collected from the web. Training uses standard cross-entropy over
token sequences. The dataset was not manually curated; transcripts were obtained
automatically (weak supervision), which accounts for the model's breadth as well
as its sensitivity to low-resource languages and strong accents.

## References

- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv:2212.04356. https://arxiv.org/abs/2212.04356
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017. https://arxiv.org/abs/1706.03762
- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL 2016. https://arxiv.org/abs/1508.07909
