from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path
import tempfile

import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement


# ============================================================
# UPDATED: Use final Tamil multilingual model
# ============================================================
CHECKPOINT_DIR = r"J:\Hackathons\DEVSOC'26\checkpoints\metricgan_tamil_augmented"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.ckpt")

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# UPDATED: Title reflects multilingual capability
app = FastAPI(title="Multilingual MetricGAN+ Enhancement API (Malayalam/Hindi/Tamil)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

enhancer = None


@app.on_event("startup")
def load_model():
    """
    Load the final multilingual model (trained on Malayalam → Hindi → Tamil)
    """
    global enhancer
    
    print("="*60)
    print("Loading Multilingual MetricGAN+ Model")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    
    # Load base model
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
        run_opts={"device": DEVICE},
    )
    
    # Load your fine-tuned weights
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        enhancer.mods.load_state_dict(checkpoint["model_state_dict"])
        enhancer.mods.enhance_model.eval()
        
        print(f"✓ Model loaded successfully!")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f} dB")
        if 'training_history' in checkpoint:
            print(f"  Training history: {checkpoint['training_history']}")
        else:
            print(f"  Training history: Malayalam → Hindi → Tamil")
    else:
        print(f"⚠️  WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
        print(f"   Using base MetricGAN+ model instead.")
    
    print("="*60)
    print("✓ API Ready!")
    print("="*60)


@app.get("/")
def root():
    """
    Health check endpoint
    """
    return {
        "status": "online",
        "model": "MetricGAN+ Multilingual",
        "languages": ["Malayalam", "Hindi", "Tamil"],
        "device": DEVICE,
        "version": "v1.0.0"
    }


@app.post("/enhance")
async def enhance_audio(file: UploadFile = File(...)):
    """
    Upload a noisy WAV file; returns enhanced WAV bytes.
    
    Supports: Malayalam, Hindi, Tamil (and likely other Indian languages)
    """
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/wave"]:
        return {"error": "Please upload a WAV file (audio/wav)."}

    # Work entirely in a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.wav"
        output_path = tmpdir_path / "output_enhanced.wav"

        # Save uploaded file
        audio_bytes = await file.read()
        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        # Enhance using multilingual model
        enhanced_waveform = enhancer.enhance_file(str(input_path))
        if enhanced_waveform.dim() == 1:
            enhanced_waveform = enhanced_waveform.unsqueeze(0)
        torchaudio.save(str(output_path), enhanced_waveform.cpu(), SAMPLE_RATE)

        # Read bytes BEFORE temp dir is deleted
        with open(output_path, "rb") as f:
            out_bytes = f.read()

    # Return raw WAV bytes
    return Response(
        content=out_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="enhanced.wav"'},
    )


@app.get("/model-info")
def model_info():
    """
    Get detailed information about the loaded model
    """
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        return {
            "checkpoint_path": CHECKPOINT_PATH,
            "epoch": checkpoint.get('epoch', 'N/A'),
            "train_loss": checkpoint.get('train_loss', 'N/A'),
            "val_loss": checkpoint.get('val_loss', 'N/A'),
            "training_lineage": "Malayalam → Hindi-English → Hindi-Augmented → Tamil",
            "languages_supported": ["Malayalam", "Hindi", "Tamil"],
            "sample_rate": SAMPLE_RATE,
            "device": DEVICE,
        }
    else:
        return {
            "error": "Checkpoint not found",
            "checkpoint_path": CHECKPOINT_PATH,
            "status": "Using base MetricGAN+ model"
        }


if __name__ == "__main__":
    print("\n🚀 Starting Multilingual Speech Enhancement API...")
    print(f"📍 Model: {CHECKPOINT_PATH}")
    print(f"🌐 Server: http://localhost:8000")
    print(f"📖 Docs: http://localhost:8000/docs\n")
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
