#!/bin/bash

# Script to set up the CaImAn Analysis GitHub repository
# Run this script from the project directory

echo "Setting up Git repository for CaImAn Analysis..."

# Configure git
git config user.name "HamedHaidari"
git config user.email "hamedh343@gmail.com"

# Add files to git (excluding data, environments, and temporary files)
echo "Adding project files to git..."
git add .gitignore
git add README.md
git add requirements.txt
git add cnmf_st_analysis.py
git add convert_lif_to_tif.py
git add components/
git add notebooks/
git add video_overview.ipynb
git add CaImAn-Helena.code-workspace

# Check status
echo "Git status:"
git status

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: CaImAn CNMF Analysis with SciPy compatibility fix

- Streamlit web interface for calcium imaging analysis
- Complete CNMF pipeline with CaImAn integration  
- SciPy 1.15+ compatibility fix for deconvolution
- Support for .lif and .tif file formats
- Interactive visualization with Bokeh
- Component evaluation and quality filtering
- Export capabilities for analysis results

Features:
- Fixes scipy.linalg.toeplitz compatibility issues
- Optimized parameters for 512x512 calcium imaging data
- Multiprocessing support for large datasets
- Comprehensive error handling and logging"

echo "Repository setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub named 'caiman-cnmf-analysis'"
echo "2. Add the GitHub remote:"
echo "   git remote add origin https://github.com/HamedHaidari/caiman-cnmf-analysis.git"
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
