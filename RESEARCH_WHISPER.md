# The Whisper Model

Background notes on the speech-recognition model whispr is built around. For
installation and usage, see the [README](README.md); this document covers the
architecture, the decoding procedure, and the model's known failure modes in
more depth.

[Whisper](https://openai.com/research/whisper) is an automatic speech
recognition (ASR) model published by OpenAI in September 2022
([Radford et al., arXiv:2212.04356](https://arxiv.org/abs/2212.04356)). It is a
sequence-to-sequence model based on a standard **encoder-decoder Transformer**,
trained end to end with weak supervision on a large-scale multilingual dataset.
Its distinguishing property is **robustness**: because it was trained on a very
broad, noisy distribution of web audio rather than one clean benchmark, it
generalises to new domains zero-shot, without any dataset-specific fine-tuning.

## Audio preprocessing

Raw audio is transformed into a spectrogram before it ever reaches the network:

1. Decoded and resampled to **16 kHz mono** (whispr does this with ffmpeg).
2. A short-time Fourier transform is computed with a **25 ms Hann window**
   (400 samples) and a **10 ms hop** (160 samples).
3. The power spectrum is projected onto an **80-channel mel filterbank**
   (`large-v3` uses **128**), then converted to log scale, clamped to an 8 dB
   dynamic range, and rescaled to roughly `[-1, 1]`.
4. The stream is processed in **30-second windows** of exactly **3000 frames**
   (a shorter tail is zero-padded to 30 s).

The resulting `80 x 3000` matrix (`128 x 3000` on `large-v3`) is the encoder
input. The fixed 30-second window is a hard architectural constant: every
inference, however short or long the audio, is framed as one or more 30 s
windows.

## Encoder

The encoder maps the spectrogram to a sequence of contextual acoustic features:

- A **convolutional stem** of two 1-D convolutions (kernel size 3, GELU
  activation). The second convolution uses **stride 2**, halving the time axis
  from 3000 to **1500 frames**.
- **Sinusoidal positional embeddings** are added to the 1500-frame sequence.
- A stack of **pre-norm Transformer blocks**, each with multi-head
  self-attention and a 4x-width MLP, followed by a final layer norm.

The encoder runs **once per 30 s window**; its output is cached and attended to
repeatedly by the decoder.

## Decoder

The decoder is an autoregressive language model over text tokens that
cross-attends to the encoder output:

- Tokenisation is **byte-level BPE** (the GPT-2 vocabulary, extended for the
  multilingual models to **51,865 tokens**; English-only models use 51,864).
- Token embeddings use **learned** positional embeddings, and the input and
  output embedding matrices are **tied**.
- Each block has masked self-attention, cross-attention to the encoder, and an
  MLP.

Generation is steered by a structured prefix of **special tokens** rather than
by task-specific heads:

- `<|startoftranscript|>` : opens the sequence.
- `<|en|>`, `<|fr|>`, ... : the target language (one of 99), or omitted to let
  the model detect it.
- `<|transcribe|>` or `<|translate|>` : transcribe in the source language, or
  translate the speech into English.
- `<|notimestamps|>` : if present, emit text only; if absent, the model
  interleaves timestamp tokens (below).
- `<|startofprev|>` : optionally precedes the prompt, used to feed the previous
  window's text back in as context.

The same weights therefore perform multilingual transcription, X-to-English
translation, language identification, and voice-activity detection, selected
purely by the prompt.

## Decoding strategy

Getting a usable transcript out of the decoder is more than a single greedy
pass. Whisper wraps generation in a set of heuristics that whispr's backends
inherit:

- **Search.** Greedy decoding at temperature 0, or beam search (typically beam
  size 5) for higher quality.
- **Temperature fallback.** A window is decoded at temperature 0 first. If its
  result fails a quality gate, the temperature is bumped in steps of 0.2 up to
  1.0 and the window is re-decoded.
- **Compression-ratio gate.** The gzip compression ratio of the text is
  checked against a threshold (~2.4). A very compressible output signals a
  **repetition loop** and triggers fallback.
- **Average-logprob gate.** If the mean token log-probability is below ~-1.0
  the decode is treated as failed.
- **No-speech gate.** The probability mass on the `<|nospeech|>` token
  (threshold ~0.6) marks a window as silence and suppresses output.
- **Context conditioning.** By default the previous window's text is fed back
  as a prompt for coherence, but this is dropped when temperature fallback
  kicks in, because a bad prompt can otherwise propagate a repetition loop
  across windows.

These gates are exactly why the same audio can produce empty output, a clean
transcript, or a runaway repetition depending on conditions: they are the model
trying, and sometimes failing, to police its own confidence.

## Timestamp prediction

When timestamp mode is active, the vocabulary includes **1501 timestamp
tokens** spanning `<|0.00|>` to `<|30.00|>` at **0.02 s resolution**. The
decoder emits `<|start|> text <|end|>` triplets inline, so **segment-level
timing falls out of ordinary decoding** with no external aligner.

**Word-level** timing is a separate, post-hoc step: it reads the
**cross-attention weights** of a subset of "alignment heads" and runs **dynamic
time warping (DTW)** between those weights and the emitted tokens to snap each
word to an audio position. This is optional and more expensive than segment
timing (whispr exposes it through `--word-timestamps` and the `stable_ts` /
`whisperx` providers).

## Long-form transcription

Audio longer than 30 s is handled by a **sliding window**. After a window is
decoded, the model **seeks forward to the last reliable timestamp** it emitted
and starts the next window there, carrying the previous text as a prompt. This
local seeking keeps windows aligned to speech boundaries, but it also means a
single mis-predicted timestamp can desynchronise the following window.

whispr sidesteps the fragile end of this by **chunking with ffmpeg** before
transcription and merging the results with corrected absolute offsets, which
keeps peak memory bounded and lets chunks run in parallel.

## Language detection

With no language specified, Whisper runs the encoder on the first 30 s, then
takes a **softmax over the 99 language tokens at the first decoder step** and
picks the most likely. This adds roughly 1-2 s of overhead and is reliable for
well-represented languages; it can misfire on very short clips, code-switching,
or heavily accented low-resource speech.

## Model family

All sizes share the same architecture and differ only in width and depth, so a
transcript's quality scales with size while the interface stays identical.

| Model | Parameters | Layers (enc/dec) | Width | Heads |
|---|---|---|---|---|
| `tiny` | 39 M | 4 | 384 | 6 |
| `base` | 74 M | 6 | 512 | 8 |
| `small` | 244 M | 12 | 768 | 12 |
| `medium` | 769 M | 24 | 1024 | 16 |
| `large` (v1/v2/v3) | 1550 M | 32 | 1280 | 20 |

Notes:

- **`.en` variants** (`tiny.en` ... `medium.en`) are trained on English only and
  are more accurate for English at a given size. There is no English-only
  `large`.
- **`large-v2`** is a re-trained `large-v1` (more epochs, regularisation) at the
  same size; it is the default "best" in many toolchains.
- **`large-v3`** switches the front-end to **128 mel bins**, adds Cantonese, and
  is trained on far more data (about 1 M hours of weakly-labelled plus 4 M hours
  of pseudo-labelled audio). It is more accurate but slightly more prone to
  hallucination on non-speech.

## Training

Whisper `large-v1`/`v2` were trained on **680,000 hours** of audio paired with
transcripts collected from the web: about **438k hours of English**, **117k
hours across 96 other languages**, and **125k hours of X-to-English
translation** pairs. Supervision is **weak**: transcripts were harvested
automatically and only lightly filtered (heuristics remove machine-generated
captions and gross audio/text mismatches). The objective is plain cross-entropy
over the token sequence.

The bet behind that design is scale over cleanliness: a large, messy, diverse
dataset buys **robustness** that a smaller, cleaner one does not. The cost is
that the model inherits the web's imbalance, which is why quality drops sharply
for low-resource languages.

## Evaluation and robustness

Whisper is evaluated **zero-shot**: it is never fine-tuned on the benchmark it
is scored against. On that footing it does not always beat models trained
directly on a given test set, but it shows **much lower error variance across
datasets**, approaching human-level robustness to distribution shift (accents,
background noise, technical vocabulary, recording conditions). Robustness, not a
single headline WER, is the model's real contribution.

## Known limitations

- **Hallucination.** On silence, music, or non-speech the model can emit
  confident but fabricated text; `large-v3` is somewhat more susceptible.
- **Repetition loops.** Decoding can get stuck repeating a phrase; the
  compression-ratio gate exists specifically to catch this.
- **Timestamp drift.** Segment timestamps are approximate, and one bad
  timestamp can desynchronise long-form seeking.
- **Low-resource languages.** Accuracy tracks how much of a language appeared in
  training; the long tail is weak.
- **Fixed 30 s frame.** Everything is forced into 30 s windows, so long audio
  depends on the seeking/chunking layer around the model, not the model alone.

## References

- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv:2212.04356. https://arxiv.org/abs/2212.04356
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017. https://arxiv.org/abs/1706.03762
- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL 2016. https://arxiv.org/abs/1508.07909
- OpenAI. *Whisper* (reference implementation, model card, and `large-v3` notes). https://github.com/openai/whisper
