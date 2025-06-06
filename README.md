# CaImAn CNMF Analysis with Streamlit

A Streamlit web application for Constrained Non-negative Matrix Factorization (CNMF) analysis of calcium imaging data using CaImAn.

## Features

- **Interactive Streamlit Interface**: Web-based GUI for calcium imaging analysis
- **CaImAn Integration**: Full CNMF pipeline with parameter optimization
- **SciPy Compatibility Fix**: Resolves SciPy 1.15+ compatibility issues with CaImAn deconvolution
- **Data Format Support**: Handles .lif and .tif files from calcium imaging experiments
- **Component Evaluation**: Quality assessment and filtering of detected neural components
- **Visualization**: Interactive plots using Bokeh for data exploration
- **Export Capabilities**: Download results in various formats (HDF5, traces, ROI files)

## Installation

### Prerequisites

1. **Conda Environment**: Create a dedicated conda environment with CaImAn
```bash
conda create -n caiman python=3.11
conda activate caiman
conda install -c conda-forge caiman
```

2. **Additional Dependencies**:
```bash
pip install streamlit bokeh holoviews
```

### Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Caiman_Analysis
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Create data directory:
```bash
mkdir -p data
```

## Usage

### Running the Application

1. Activate the CaImAn environment:
```bash
conda activate caiman
```

2. Start the Streamlit application:
```bash
streamlit run cnmf_st_analysis.py
```

3. Open your browser to `http://localhost:8501`

### Analysis Workflow

1. **Data Loading**: 
   - Place your .lif or .tif files in the `data/` directory
   - Select files through the web interface

2. **Parameter Configuration**:
   - Configure CNMF parameters (patch size, components per patch, etc.)
   - Parameters are optimized for 512×512 pixel images at 28Hz

3. **CNMF Analysis**:
   - Click "Run CNMF" to start component detection
   - Monitor progress and view diagnostic information

4. **Component Evaluation**:
   - Evaluate component quality using automated metrics
   - Filter components based on SNR, spatial correlation, etc.

5. **Results Export**:
   - Download calcium traces, spatial components, and analysis results
   - Export in formats compatible with further analysis

## Key Components

### Core Files

- **`cnmf_st_analysis.py`**: Main Streamlit application
- **`components/`**: Modular components for different analysis steps
  - `utils.py`: Data loading and file handling utilities
  - `parameters.py`: CNMF parameter configuration interface
  - `general.py`: Visualization and plotting functions
  - `scipy_fix.py`: **SciPy compatibility fix for CaImAn deconvolution**

### SciPy Compatibility Fix

This project includes a critical fix for SciPy 1.15+ compatibility issues with CaImAn's deconvolution module:

- **Problem**: `scipy.linalg.toeplitz` no longer auto-flattens 2D inputs
- **Solution**: Monkey patch that ensures 1D inputs to `toeplitz` function
- **Impact**: Enables successful CNMF fitting that previously returned `None`

### Optimized Parameters

Default parameters are optimized for:
- **Image Size**: 512×512 pixels
- **Frame Rate**: 28Hz calcium imaging
- **Patch Size**: 25 pixels (reduced from default 40)
- **Components**: 4 per patch (reduced from default 5)
- **Neuron Size**: 3×3 pixels (reduced from default 5×5)

## Data Structure

```
Caiman_Analysis/
├── cnmf_st_analysis.py          # Main application
├── components/                   # Modular components
│   ├── scipy_fix.py             # SciPy compatibility fix
│   ├── utils.py                 # Data utilities
│   ├── parameters.py            # Parameter configuration
│   └── general.py               # Visualization functions
├── data/                        # Data directory (excluded from git)
│   ├── *.lif                    # Original microscopy files
│   └── *.tif                    # Converted imaging data
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Technical Details

### Environment Requirements

- **Python**: 3.11
- **CaImAn**: 1.12.1
- **SciPy**: 1.15.2 (with compatibility fix)
- **NumPy**: Compatible with CaImAn
- **Streamlit**: Latest version

### Memory Requirements

- **Large Datasets**: Supports datasets with 8000+ frames
- **Memory Mapping**: Uses numpy.memmap for efficient large file handling
- **Multiprocessing**: Supports parallel processing with configurable worker processes

### Known Issues & Solutions

1. **SciPy Compatibility**: Resolved with included monkey patch
2. **Memory Usage**: Use memory mapping for large datasets
3. **Cluster Management**: Automatic fallback to single-process mode if needed

## Development

### Contributing

1. Follow the existing code structure with modular components
2. Test changes with representative calcium imaging datasets
3. Ensure compatibility with CaImAn's parameter structure
4. Update documentation for any new features

### Testing

- Test with various image sizes (512×512, 1024×1024)
- Verify with different frame rates (4Hz, 28Hz)
- Check memory usage with large datasets (>5GB)

## Citation

If you use this tool in your research, please cite:

1. **CaImAn**: Giovannucci, A. et al. CaImAn an open source tool for scalable calcium imaging data analysis. eLife 8, e38173 (2019).
2. **This Project**: [Add your citation information]

## License

[Add your license information]

## Contact

[Add your contact information]

---

**Note**: This project includes significant improvements to the standard CaImAn workflow, particularly the SciPy compatibility fix that resolves critical issues with modern Python environments.
