# VerbalVector: Local Model Distillation for Presentation Feedback

VerbalVector is a project that uses model distillation to create a lightweight, efficient speech analytics model for providing presentation feedback. The system analyzes presentation recordings and generates expert-level feedback on delivery and content, running entirely on local hardware.

## Architecture

The project implements a teacher-student model distillation approach using locally hosted models via [Ollama](https://ollama.com/):

- **Teacher Model**: **Gemma 2 9B Instruct** (`gemma2:9b`) hosted locally via Ollama, providing high-quality expert feedback.
- **Student Model**: **Gemma 2 2B Instruct** (`google/gemma-2-2b-it`), distilled from the teacher model for efficient local inference using Hugging Face Transformers.

## Key Features

- Presentation feature extraction (audio + transcript analysis).
- Expert-level feedback generation using a local Gemma 2 9B model.
- Model distillation to create a smaller, faster local Gemma 2 2B feedback model.
- Comprehensive evaluation framework comparing the student model against the local teacher.
- Command-line interface for running the full workflow.

## Project Structure

```
VerbalVector/
├── data/
│   ├── raw/           # Raw audio recordings (input)
│   ├── processed/     # Processed feature files (*_features.json)
│   └── feedback/      # Teacher model feedback (*_feedback.json)
├── models/
│   └── distilled_feedback/ # Trained student model assets
├── evaluation_results/  # Output directory for model evaluation
│   ├── plots/         # Evaluation plots
│   ├── aggregated_metrics.json
│   └── evaluation_results.json
├── src/
│   ├── data/          # Data collection utilities
│   ├── features/      # Feature extraction logic
│   ├── models/        # Teacher (Ollama) and student (Transformers) implementation
│   └── evaluation/    # Model evaluation utilities
├── main.py            # Main CLI script
├── requirements.txt   # Project dependencies
├── README.md          # This file
└── .gitignore         # Git ignore file
```

## Getting Started

### Requirements

- Python 3.8+
- **Ollama**: Install Ollama from [ollama.com](https://ollama.com/) and ensure it is running.
- **FFmpeg**: Required for audio processing (`pydub`). Install via your system package manager (e.g., `brew install ffmpeg` on macOS, `sudo apt update && sudo apt install ffmpeg` on Debian/Ubuntu).
- Dependencies listed in `requirements.txt`.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/VerbalVector.git # Replace with your repo URL
cd VerbalVector

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull Ollama models (Run this in your terminal)
ollama pull gemma2:9b  # Teacher model
# ollama pull gemma2:2b  # Optional: if you want to run the base 2B model via Ollama too
```

### Running the Workflow (via `main.py`)

The `main.py` script provides a command-line interface. Ensure Ollama is running before executing commands that require the teacher model (`teacher`, `evaluate`).

```bash
# Optional: Ensure project directories exist
python main.py setup

# 1. Extract features from audio files in data/raw/
# Place your .mp3, .wav, etc. files in data/raw/ first
python main.py extract --audio-dir data/raw --output-dir data/processed

# 2. Generate teacher feedback using local Gemma 2 9B
# Requires Ollama running with 'gemma2:9b' pulled
python main.py teacher --feature-dir data/processed --output-dir data/feedback --teacher-model gemma2:9b

# 3. Train the student model (Gemma 2 2B)
python main.py train --feature-dir data/processed --feedback-dir data/feedback --output-dir models/distilled_feedback --student-model google/gemma-2-2b-it --epochs 5

# 4. Evaluate the trained student model against the local teacher
# Requires Ollama running with 'gemma2:9b' pulled
python main.py evaluate --student-model-path models/distilled_feedback --feature-dir data/processed --output-dir evaluation_results --student-model google/gemma-2-2b-it --teacher-model gemma2:9b

# 5. Run inference with the trained student model on a single feature file
python main.py infer --student-model-path models/distilled_feedback --feature-path data/processed/your_presentation_features.json --student-model google/gemma-2-2b-it
```

*(Replace `your_presentation_features.json` with an actual file name)*

## Model Details

### Teacher Model: Gemma 2 9B Instruct (`gemma2:9b` via Ollama)

- Parameter Count: 9 billion
- Served locally using Ollama for cost-free execution.
- Provides high-quality feedback for distillation.

### Student Model: Gemma 2 2B Instruct (`google/gemma-2-2b-it`)

- Parameter Count: ~2.6 billion
- Fine-tuned using Hugging Face Transformers based on teacher feedback.
- Optimized for efficient local inference (CPU or modest GPU).
- Can be further quantized to 4-bit precision using the built-in optimization step.

## License

MIT License

## Acknowledgements

- Google for providing the Gemma models.
- The Ollama team for enabling easy local LLM execution.
- Hugging Face for the Transformers library. 