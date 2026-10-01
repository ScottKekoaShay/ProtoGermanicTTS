# Proto-Germanic TTS: a Piper voice for reconstructed Proto-Germanic

A neural text-to-speech voice that speaks Proto-Germanic from IPA.

No prior TTS model exists for Proto-Germanic (as far as I know): there are no recordings
of the language. This voice is a fine-tune of a Castilian Spanish Piper model
on a small corpus of Proto-Germanic citation forms read aloud by one speaker,
driven directly by phoneme input.

It was built for the audio layer of *Wurdahuzdą: Proto-Germanic dictionary*
(Scott K. Shay, 2026 https://protogermanic.org), where it supplies a spoken pronunciation
for each of ~5,300 headwords.

---

## What's here

| file | size | notes |
|---|---|---|
| [pgmc.onnx](https://github.com/ScottKekoaShay/ProtoGermanicTTS/releases/download/v1.0/pgmc.onnx) | ~64 MB | the voice, Piper "medium" quality, 22.05 kHz |
| [pgmc.onnx.json](https://github.com/ScottKekoaShay/ProtoGermanicTTS/releases/download/v1.0/pgmc.onnx.json) | ~5 KB | phoneme map, audio config, inference defaults |
| [epoch.7219-step.1798080.ckpt](https://github.com/ScottKekoaShay/ProtoGermanicTTS/releases/download/v1.0/epoch.7219-step.1798080.ckpt) | ~826 MB | checkpoint to refine the model further (not needed to run this) |
| `pgmc_tts.py` | small | minimal inference example (ONNX Runtime + numpy) |
| `normalize.py` | small | **IPA → model input. Required.** See below. |

---

## Quick start

```bash
pip install onnxruntime numpy soundfile
python pgmc_tts.py "ˈhun.dɑz" out.wav
```

---

## The normalization step is not optional

The model was trained on a specific normalized form of IPA. Feeding it raw
dictionary transcriptions produces audible errors -- not silence or an error
message, but confident mispronunciation. `normalize.py` applies a transform
that uses eSpeak's conventions:

| input | sent to model | why |
|---|---|---|
| `ˈxun.dɑz` | `xˈundɑz` | **stress moves to just before its vowel**. eSpeak's convention, which the training labels follow. A stress mark before a consonant is out of distribution. |
| `ˈxun.dɑz` | `xˈundɑz` | **syllable dots are dropped**. `.` maps to the *punctuation* period, so a three-syllable word would render as three sentences. |
| `ˈɑi̯.nɑz` | `ˈɑinɑz` | **U+032F (non-syllabic) is stripped**. It has an id in the map but was never in training, so its embedding is untrained -- this one inserts whole spurious syllables. |
| `ˈɑ.xʷɔː` | `ˈɑxwɔː` | **ʷ → w**. Labiovelars are two segments. |
| `ĩ` `ũ` | `i`+U+0303 | **nasal vowels are decomposed** to base + combining tilde (U+0303, id 141). The precomposed forms have no id. |

Normalization occurs transparently with the attached pgmc_tts.py script,
but you cannot skip it if you modify the process.

Phoneme ids follow Piper's convention: `BOS, PAD, p, PAD, p, PAD, …, EOS`.
The `PAD` immediately after `BOS` matters -- omitting it shifts every token by
one position and the duration predictor inserts or drops syllables.

---

## Phoneme inventory

Trained on the Proto-Germanic inventory as transcribed in the dictionary:

```
p t k b d ɡ   f θ s x   β ð ɣ ɸ z   m n ŋ   l r j w
ɑ e i u ɔ ɛ   ɑː eː iː uː ɔː ɛː
ɑ̃ ẽ ĩ ɔ̃ ũ  (base + U+0303)       ɔːː  (overlong)
ˈ ˌ ː
```

Vowel length is three-way where the reconstruction has it: short, long, and
overlong (e.g. `ɔːː`, the trimoraic vowels *ô*/*ê* ). Nasal vowels are
contrastive. Anything outside this set is either dropped or renders
unpredictably -- check before you rely on it.

---

## How it was built

**Base model.** `es_ES-davefx-medium` from
[rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints)
(`epoch=5629-step=1605020`). Castilian Spanish was chosen on a phonological
audit rather than by convenience: it is the only widely-available voice whose
inventory already contains β, ð, ɣ, θ and x, all of which are frequent in
Proto-Germanic and all of which are hard to learn from a small corpus.

**Choosing what to record.** With ~5,300 headwords and one speaker, recording
everything was not realistic. Instead, the corpus was reduced to its *diphone*
inventory (based on singular nominative forms for nouns and adjectitves and on infitinitive
forms for verbs) -- adjacent phone pairs, since transitions are where synthesis fails --
and a greedy set cover selected the smallest word list covering all of them:

```
5,320 headwords  →  760 distinct diphones
324 words        →  100% diphone coverage   (~7.4 min of audio)
```

Coverage saturates far faster than intuition suggests. A second pass added 353
more words chosen to reinforce the thinnest diphones -- 468 of the 760 had
exactly one exemplar after phase one, and one example is not enough to learn a
transition from. 

**Recording.** One speaker, phone microphone, continuous takes of ~30 words
read from a prompter with a deliberate pause between each, segmented
automatically on silence with a count check against the expected word list.
Consistency mattered more than equipment: same room, same position, short
sessions.

**Training.** Fine-tuned for approximately 1000 epochs on a single GPU, with
`--phoneme-type text` so the trainer treated `metadata.csv` as phonemes
directly rather than running espeak over it -- which is what made it possible
to train a language espeak had never heard of. The phoneme id map was
remapped to Piper's standard 256-symbol table so the pretrained embeddings
carried over.

**Post-processing.** Raw output had a thin, slightly metallic quality
characteristic of a vocoder trained on limited data. Spectral analysis against
the speaker's own recordings showed the synthesis running ~9 dB hot at
700–1500 Hz and ~6 dB hot at 5–7 kHz. Running the output through a speech
restoration model (Adobe Podcast Enhance) corrected this and is strongly
recommended -- this processed audio was then further used to refine the 
 model for nearly 600 more ephochs. 

---

## Known limitations

- **22.05 kHz.** Piper "medium" ceiling. Fricative detail is limited.
- **Trained on one voice, ~3.5 hours of audio, approx 15 mins real 
  audio the rest artificial.
 - ** Expect artifacts on long or unusual words.
- **Citation forms only.** Every training item is an isolated word with fixed
  initial stress. Connected speech, questions and phrasing are outside its
  experience; the few multi-word entries in the training data number in single
  digits.
- **Length and nasality are its strengths.** Reviewers found the long/overlong
  vowel contrast and the nasal vowels more consistent than a human reader of a
  language that lacks both.

---

## License

The model weights are a derivative of a Piper checkpoint; this software is MIT-licensed.

## Citation

> Shay, Scott K. 2026. *Wurdahuzdą: Proto-Germanic dictionary*. 

