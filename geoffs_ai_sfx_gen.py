import torch
import platform
import sys
import transformers
import diffusers
import numpy as np
import matplotlib.pyplot as plt
from diffusers import AudioLDM2Pipeline
from IPython.display import Audio, display
# from scipy.io.wavfile import write # Added for saving audio (see notes)

# --- Environment Check and Setup ---
print(f"Python:        {sys.version.split()[0]}")
print(f"Platform:      {platform.platform()}")
print(f"Torch:         {torch.__version__}")
print(f"Transformers:  {transformers.__version__}")
print(f"Diffusers:     {diffusers.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print(f"CUDA available:{torch.cuda.is_available()}")

# --- Load Pipeline ---
REPO_ID = "cvssp/audioldm2"
# Use half precision on GPU; fall back to float32 on CPU
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"\nLoading AudioLDM2 model '{REPO_ID}'...")
pipe = AudioLDM2Pipeline.from_pretrained(
    REPO_ID,
    torch_dtype=torch_dtype
).to(device)
print("Model loaded successfully.")

# Sanity checks
print("text_encoder:", type(pipe.text_encoder).__name__)
print("tokenizer:", type(pipe.tokenizer).__name__)
print("has CLAP:", hasattr(pipe, "clap_model"))

# --- Generation Function ---
def generate_sfx(
    prompt: str,
    negative_prompt: str = "Low quality.",
    length_s: float = 10.24,
    steps: int = 100,
    guidance: float = 7.0,
    seed: int = 0,
):
    """
    Generates audio using the AudioLDM2 model based on the provided prompt.

    Args:
        prompt (str): The text prompt describing the desired sound.
        negative_prompt (str): Prompt describing unwanted qualities (e.g., noise).
        length_s (float): Duration of the generated audio in seconds.
        steps (int): Number of inference steps (higher is generally better quality).
        guidance (float): Classifier-free guidance scale.
        seed (int): Seed for reproducible generation.

    Returns:
        np.ndarray: The generated audio as a float32 numpy array (16 kHz).
    """
    g = torch.Generator(device).manual_seed(seed)
    out = pipe(
        prompt,
        negative_prompt=negative_prompt,
        audio_length_in_s=length_s,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
    )
    # AudioLDM2 generates audio at 16 kHz
    audio = out.audios[0]
    return audio

# --- Main Execution Block ---
if __name__ == "__main__":
    PROMPT = "A Busy Coffee shop, relaxing vibe" # <--- Change your prompt here!

    print(f"\nGenerating SFX for prompt: '{PROMPT}'")
    audio = generate_sfx(PROMPT)
    print("Generation complete.")

    # Inline audio player (Only works in Jupyter/Colab environments)
    try:
        display(Audio(audio, rate=16000))
    except NameError:
        print("Note: Audio playback widget requires running in a Jupyter or Colab notebook.")
        # If running locally, you would typically save the file here:
        # write("output.wav", 16000, audio)

    # Waveform plot
    plt.figure(figsize=(10, 2.5))
    plt.title(f"Generated waveform (16 kHz) for: {PROMPT}")
    plt.plot(np.arange(len(audio)) / 16000.0, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
