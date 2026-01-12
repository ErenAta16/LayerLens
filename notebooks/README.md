# LayerLens Notebooks

This directory contains Jupyter notebooks for interactive demos and tutorials.

## Quick Start

### Google Colab
Click the badge below to open the quick start notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErenAta16/LayerLens/blob/main/notebooks/colab_quick_start.ipynb)

### Available Notebooks

#### `colab_quick_start.ipynb`
**Recommended for first-time users**

A beginner-friendly introduction to LayerLens on Google Colab:
- ✅ One-click installation
- ✅ GPU verification
- ✅ Simple BERT example
- ✅ Results visualization
- ⏱️ ~5 minutes to complete

**Topics covered:**
- Installing LayerLens on Colab
- Defining model specifications
- Configuring profiling and optimization
- Running the pipeline
- Interpreting results

## Local Usage

To run these notebooks locally:

```bash
# Install Jupyter
pip install jupyter

# Install LayerLens with demo dependencies
pip install -e ".[demo]"

# Start Jupyter
jupyter notebook notebooks/
```

## Troubleshooting

If you encounter issues on Colab, see [COLAB_TROUBLESHOOT.md](../COLAB_TROUBLESHOOT.md) for solutions.

## Contributing

Want to add more notebooks? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

