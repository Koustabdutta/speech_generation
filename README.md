# Speech VAE — Compression, Synthesis & Anomaly Detection

**Professional project title:** *Speech VAE — Robust Mel-Spectrogram VAE for Compression, Synthesis & Anomaly Detection*

---

## Overview

This repository contains a self‑contained PyTorch project that trains a convolutional Variational Autoencoder (VAE) on mel‑spectrograms for speech compression, synthesis and anomaly detection. The pipeline uses `librosa` + `soundfile` for audio I/O (no `torchcodec` dependency), produces extensive evaluation metrics and saves a suite of visualization images and audio examples.

The code is suitable for research prototypes, demos, or as a starting point for deploying lightweight audio anomaly detection systems.

---

## Key features

* Loads audio files recursively (WAV/FLAC/MP3) using `librosa`.
* Converts audio to mel spectrograms, normalizes per sample and trains a convolutional VAE.
* Reconstructs audio from model outputs using `librosa` inverse-mel (Griffin‑Lim) and saves `.wav` samples.
* Performs anomaly detection using reconstruction error and computes classification metrics (ROC‑AUC, precision/recall, confusion matrix).
* Produces many visualizations: training curves, spectrogram comparisons, latent space (t‑SNE / PCA), waveform & spectrogram plots, and anomaly analysis charts.
* Notebook‑friendly playback (Jupyter) for listening to original & reconstructed audio.

---

## Repository structure

```
/ (repo root)
├─ README.md               # This file
├─ speech_vae.py           # Main script (training / eval / plots)
├─ requirements.txt        # Python deps (suggested)
├─ /librispeech_data       # Put your audio files here (wav/flac/mp3) or LibriSpeech dev-clean
└─ /output                 # Generated models / plots / audio saved here
   ├─ /models
   ├─ /plots
   │  ├─ training/
   │  ├─ reconstruction/
   │  ├─ latent/
   │  └─ anomaly/
   └─ /audio
```

> Note: The script name may differ; replace `speech_vae.py` with your actual file name if needed.

---

## Requirements

Tested with Python 3.8+ (works on 3.10/3.11). Install the packages below:

```bash
pip install torch numpy librosa soundfile matplotlib seaborn scikit-learn tqdm
```

> You do **not** need `torchaudio`/`torchcodec` for this repository — the pipeline intentionally uses `librosa` to avoid common Windows FFmpeg/torchcodec issues.

---

## Quick start

1. **Prepare data** — place audio files (WAV/FLAC/MP3) under `./librispeech_data`. You can download LibriSpeech `dev-clean` manually and extract WAV/FLAC files into that folder.

2. **Adjust configuration** — open the script and edit `Config` fields at the top:

* `DATA_PATH` — path to audio folder
* `NUM_SAMPLES` — how many files to use (use small number for quick tests)
* `EPOCHS`, `BATCH_SIZE`, `LATENT_DIM` etc. for model/training control

3. **Run training & evaluation**:

```bash
python speech_vae.py
```

The script will:

* load data, split into train/val/test
* train the VAE, saving the best model to `output/models/best_vae_model.pth`
* run evaluation and anomaly detection
* save visualizations to `output/plots` and audio examples to `output/audio`

---

## What you will get (outputs)

* `output/models/best_vae_model.pth` — saved model checkpoint
* `output/plots/` — many PNG visualizations (training curves, spectrogram comparisons, latent visualizations, anomaly analysis)
* `output/audio/` — saved `original_*.wav` and `reconstructed_*.wav` examples

---

## How to listen to generated audio

* If you run the script in a Jupyter notebook, the script will attempt in‑notebook playback for the first pair of audio using IPython's `Audio` (if available).
* Otherwise open `output/audio/reconstructed_1.wav` (and `original_1.wav`) in any audio player (VLC, Windows Media Player, Audacity).

---

## Troubleshooting & tips

* **No audio files found**: Make sure `DATA_PATH` points to a directory containing audio files (extensions: `.wav`, `.flac`, `.mp3`, `.m4a`, `.aac`).
* **Too slow on CPU**: Reduce `NUM_SAMPLES`, `EPOCHS`, and `HIDDEN_DIMS` in `Config` for faster iteration.
* **TorchCodec / torchaudio errors**: This project avoids `torchaudio.load` because `torchcodec` can fail on some Windows setups. Using `librosa` avoids that dependency.
* **Better audio quality**: Increase `n_iter` in `librosa.feature.inverse.mel_to_audio` inside the `AudioReconstructor` (longer Griffin‑Lim iterations give better audio but are slower).

---

## Recommended experiments

* Try smaller/larger `LATENT_DIM` and measure PSNR / SSIM changes.
* Replace Griffin‑Lim with a neural vocoder (MelGAN, HiFi‑GAN) for higher-quality reconstruction.
* Use the learned latent codes for downstream classification or speaker embedding tasks.

---

## License

Include your preferred license (e.g., MIT). Add a `LICENSE` file to the repo.

---

## Contact

For questions or requests (readme edits, smaller model variants, notebook examples), open an issue or ping me in the repo.

---

*Generated README — modify as needed before uploading to GitHub.*
