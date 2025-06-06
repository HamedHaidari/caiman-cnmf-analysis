#!/bin/bash

echo "🚀 Setting up CaImAn CNMF Analysis for GitHub..."
echo "================================================"

# Check if we're in the right directory
if [ ! -f "cnmf_st_analysis.py" ]; then
    echo "❌ Error: Please run this script from the project directory"
    exit 1
fi

echo "✅ Project directory confirmed"

# Configure git
echo "🔧 Configuring git..."
git config user.name "HamedHaidari"
git config user.email "hamedh343@gmail.com"

# Add files
echo "📁 Adding project files to git..."
git add .gitignore
git add README.md  
git add requirements.txt
git add cnmf_st_analysis.py
git add convert_lif_to_tif.py
git add components/
git add notebooks/
git add video_overview.ipynb
git add CaImAn-Helena.code-workspace
git add GITHUB_SETUP.md

echo "📊 Git status:"
git status

echo ""
echo "🎯 Ready for commit! Run:"
echo "git commit -m 'Initial commit: CaImAn CNMF Analysis with SciPy compatibility fix'"
echo ""
echo "📖 See GITHUB_SETUP.md for complete instructions"
