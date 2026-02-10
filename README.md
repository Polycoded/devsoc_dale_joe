“We built a single AI model that takes noisy Indian speech (Hindi, Malayalam, Tamil, English with Indian accent) and outputs a clean version.
We started from a strong English speech enhancement model (MetricGAN+), and fine‑tuned it using transfer learning on Indian language datasets.
The model is served through a FastAPI backend (app.py) and a simple web UI (front.html), and we demonstrate the results using our before/after WAV files and the architecture shown in image.png.”



🧠 Model Overview
We start from MetricGAN+ (SpeechBrain), a strong speech enhancement model trained on English, and then fine‑tune it step by step on Indian language datasets using transfer learning.

Training pipeline (high level)
Base MetricGAN+ (pretrained on English speech)

Fine‑tune on Malayalam noisy/clean pairs

Fine‑tune on Hindi / English with Indian accent

Fine‑tune on additional Indic data (Hindi, Malayalam, etc.)

Export final weights as best.model

All fine‑tuning scripts and experiments are in transfer_learning.ipynb.

📦 Backend – app.py
app.py is a FastAPI application that:

Loads the final speech enhancement model (best.model)

Exposes an /enhance endpoint

Accepts a noisy WAV file upload

Returns the enhanced WAV bytes

How it works
Client uploads a .wav file to /enhance

Backend saves it temporarily

Model enhances the audio (denoising / dereverberation)

Backend sends back a cleaned .wav file

You can also add health/info endpoints (e.g. /, /model-info) to inspect model status and metadata.

🌐 Frontend – front.html
front.html is a simple web page that:

Lets the user select a WAV file

Sends it to the FastAPI /enhance endpoint

Lets the user download or play back the enhanced result

Typical flow:

Open front.html in a browser

Choose a noisy audio file (Hindi/Malayalam/English with Indian accent)

Click Enhance

Listen to or download the cleaned audio

This makes the demo highly intuitive for judges and users.

🧪 Example Audio Files
There are two .wav files included:

noisy_*.wav – original noisy recording

enhanced_*.wav – output from our fine‑tuned model

Use these for:

Quick offline comparison

Presentations and demos

Before/after listening tests

📓 Training Code – transfer_learning.ipynb
This Jupyter notebook contains the end‑to‑end training logic:

Dataset loading (noisy/clean pairs)

Resampling to 16 kHz

Padding and batching

Using MetricGAN+ from SpeechBrain as a base model

Transfer learning across Malayalam, Hindi, and Indian‑accent English

SI‑SNR‑based loss for perceptual quality

You can open this notebook to:

Reproduce training

Modify hyperparameters

Extend to new languages (e.g., Tamil, Telugu, etc.)

🖼️ Model Diagram – image.png
image.png illustrates:

The high‑level architecture of the system
(Frontend → FastAPI Backend → MetricGAN+ Model → Enhanced Audio)

Or the internal training flow (Noisy → Model → Clean)

Include this image in your slides or documentation for a quick visual explanation.

📚 Datasets Used
We used Indian language speech datasets for fine‑tuning:

IIIT Voices (Indian accented speech)
http://festvox.org/databases/iiit_voices/

Indic TTS (IIT Madras) – Indic language TTS databases
https://www.iitm.ac.in/donlab/indictts/database

From these sources, we derived:

Noisy/clean paired audio for:

Hindi

Malayalam

English with Indian accent

Tamil

Augmented noisy versions for robustness (different noise types, SNRs)

Always check and comply with each dataset’s license/usage policy before using in production.

