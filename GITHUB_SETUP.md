# GitHub Setup Guide for CaImAn CNMF Analysis

## 📋 Step-by-Step Instructions

### 1. **First, run these commands in your terminal:**

```bash
cd /home/hhaidari/Documents/CaImAn-Helena/Caiman_Analysis

# Configure git
git config user.name "HamedHaidari"
git config user.email "hamedh343@gmail.com"

# Add project files (excluding data)
git add .gitignore
git add README.md
git add requirements.txt
git add cnmf_st_analysis.py
git add convert_lif_to_tif.py
git add components/
git add notebooks/
git add video_overview.ipynb
git add CaImAn-Helena.code-workspace

# Check what's staged
git status

# Create initial commit
git commit -m "Initial commit: CaImAn CNMF Analysis with SciPy compatibility fix

- Streamlit web interface for calcium imaging analysis
- Complete CNMF pipeline with CaImAn integration  
- SciPy 1.15+ compatibility fix for deconvolution
- Support for .lif and .tif file formats
- Interactive visualization with Bokeh
- Component evaluation and quality filtering
- Export capabilities for analysis results"
```

### 2. **Create a new repository on GitHub:**

1. Go to https://github.com/HamedHaidari
2. Click the "+" button in the top right corner
3. Select "New repository"
4. **Repository name**: `caiman-cnmf-analysis`
5. **Description**: `Streamlit web application for calcium imaging analysis using CaImAn CNMF with SciPy compatibility fixes`
6. Make it **Public** (or Private if you prefer)
7. **DON'T** initialize with README (we already have one)
8. Click "Create repository"

### 3. **Connect your local repository to GitHub:**

```bash
# Add GitHub as remote origin
git remote add origin https://github.com/HamedHaidari/caiman-cnmf-analysis.git

# Rename branch to main (GitHub default)
git branch -M main

# Push to GitHub
git push -u origin main
```

### 4. **Verify everything worked:**

- Visit https://github.com/HamedHaidari/caiman-cnmf-analysis
- You should see all your code files
- Check that data/ folder is NOT present (excluded by .gitignore)
- Verify the README.md displays properly

## 🎯 What Gets Uploaded to GitHub:

✅ **Included:**
- All Python source code (.py files)
- Documentation (README.md)
- Dependencies (requirements.txt)
- Configuration files (.gitignore, workspace settings)
- Jupyter notebooks
- Components and utilities

❌ **Excluded (by .gitignore):**
- Data files (*.lif, *.tif, data/ folder)
- CaImAn generated files (*.mmap, caiman_data/)
- Virtual environments (caiman/ folder)
- Cache files (__pycache__/, *.pyc)
- Results and outputs
- Temporary files

## 🚀 Project Highlights:

Your repository will showcase:

1. **SciPy Compatibility Fix** - Solves scipy.linalg.toeplitz issues with CaImAn
2. **Streamlit Web Interface** - User-friendly GUI for calcium imaging analysis
3. **Complete CNMF Pipeline** - From data loading to component evaluation
4. **Optimized Parameters** - Tuned for 512x512 calcium imaging data
5. **Professional Documentation** - Comprehensive README with installation guide

## 💡 Repository Features:

- **Interactive Analysis**: Web-based interface for non-programmers
- **Scalable Processing**: Multiprocessing support for large datasets  
- **Quality Control**: Component evaluation and filtering
- **Export Options**: Multiple output formats for further analysis
- **Error Handling**: Robust error handling and user feedback

This creates a professional, well-documented repository that other researchers can use for their calcium imaging analysis projects!
