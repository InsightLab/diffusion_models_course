
# Diffusion Models Minicourse (SBBD 2025)

Welcome to the repository for the **Diffusion Models Minicourse** presented at the Simpósio Brasileiro de Banco de Dados (SBBD) 2025.

The **Diffusion Models Minicourse** will be presented at SBBD (Simpósio Brasileiro de Banco de Dados) 2025. This 4-hour session includes a written paper (currently awaiting publication) and a slide-based presentation.

## Materials
- Published chapter: https://books-sol.sbc.org.br/index.php/sbc/catalog/book/179
- Slides: https://docs.google.com/presentation/d/1obWOI-vrOl7_5cF3OsFMdRx_SCOhu-veGGYFOEHgj1o/edit?usp=sharing

## Repository Purpose
The main content of this repository is to provide the hands-on part of the minicourse. You will find practical Jupyter Notebooks that guide you through the implementation and experimentation with diffusion models.

## Notebooks
This repository contains three main notebooks:

1. **From Scratch: Denoising Diffusion Probabilistic Model**
	- Step-by-step implementation of a denoising diffusion probabilistic model (DDPM) from scratch.
2. **From Scratch: Score-Based Model**
	- Step-by-step implementation of a score-based generative model from scratch.
3. **Inference Implementation with Diffusers**
	- Practical usage of diffusion models using the Hugging Face `diffusers` library.

## Requirements
- **Python:** 3.13 or higher
- **GPU:** CUDA-compatible GPU recommended for faster training (CUDA 12.9 support included)
- **Memory:** At least 8GB RAM (16GB+ recommended for larger models)

## Setup Instructions

### Option 1: Using uv (Recommended)
This project uses [uv](https://docs.astral.sh/uv/) as the package manager for faster and more reliable dependency management.

1. **Install uv** (if not already installed):
   ```bash
   # On Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone this repository:**
   ```bash
   git clone https://github.com/InsightLab/diffusion_models_course.git
   cd diffusion_models_course
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Activate the environment:**
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

### Option 2: Using pip and venv
If you prefer using traditional Python tools:

1. **Clone this repository:**
   ```bash
   git clone https://github.com/InsightLab/diffusion_models_course.git
   cd diffusion_models_course
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # Activate it
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install PyTorch with CUDA support:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
   ```

4. **Install other dependencies:**
   ```bash
   pip install matplotlib seaborn tqdm
   ```

## Getting Started

### Running the Notebooks
1. **Start Jupyter:**
   ```bash
   # Make sure your environment is activated
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Open and run the notebooks:**
   - `ddpm_from_scratch.ipynb` - DDPM implementation from scratch
   - `ncsn_from_scratch.ipynb` - Score-based model from scratch  
   - `hugging_face_diffusers.ipynb` - High-level inference implementation with Diffusers

3. **Follow the instructions** in each notebook to run the code and explore diffusion models.

### Project Structure
```
diffusion_models_course/
├── models/               # Model implementations
│   ├── ddpm.py           # DDPM model architecture
│   └── ncsn.py           # NCSN model architecture
├── weights/              # Pre-trained model weights
├── samples/              # Generated sample images
├── data/                 # Dataset storage (MNIST)
├── pyproject.toml        # Project metadata and dependencies
└── *.ipynb               # Jupyter notebooks
```

### GPU Support
The project is configured to use CUDA 12.9. If you have a different CUDA version or want CPU-only execution:
- For different CUDA versions, modify the PyTorch installation command
- For CPU-only: remove the `--index-url` parameter when installing PyTorch

### Troubleshooting
- **CUDA issues:** Ensure your GPU drivers are up to date
- **Memory issues:** Reduce batch sizes in the notebooks if you encounter OOM errors
- **uv not found:** Make sure uv is properly installed and added to your PATH

## License
See the [LICENSE](LICENSE) file for details.
