👤 Author - Geoff Bremner.

🔗 Connect and See More Projects: https://linktr.ee/gbaudio

# 🤖 AI Sound Effects Generator (AudioLDM2)

This project utilizes the **AudioLDM2** diffusion model to generate high-quality sound effects (SFX) and ambient audio from simple text prompts. It is optimized for use in GPU-accelerated environments like Google Colab or local machines with NVIDIA GPUs.

## 🚀 Quick Start (Recommended)

The fastest way to get started and run the code is directly in Google Colab, where all dependencies and GPU runtime are handled automatically.

* **Launch in Google Colab:**
    # <a href="https://colab.research.google.com/drive/1BcJuhFZSw3N7UFv3O0um_PW6ysKVS-og" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

    > **Note:** Select "GPU" as the runtime type in Colab (`Runtime` -> `Change runtime type`).

---

## ⚙️ Requirements and Installation

To run this script locally on your machine, you will need Python 3.8+ and the libraries listed in `requirements.txt`. A dedicated GPU is highly recommended for generation speed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/geoffbremner/geoffs-ai-sfx-gen.git](https://github.com/geoffbremner/geoffs-ai-sfx-gen.git)
    cd geoffs-ai-sfx-gen
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### 1. Set Your Prompt

Edit the `geoffs_ai_sfx_gen.py` file and modify the `PROMPT` variable in the main execution block:

```python
    PROMPT = "A Busy Coffee shop, relaxing vibe" # <--- Change your prompt here!
