import multiprocessing
import os
import re
import sys
from pathlib import Path

import bokeh
import bokeh.plotting as bpl
import holoviews as hv
import numpy as np
import streamlit as st
from bokeh.layouts import row
from bokeh.models import Column, Row, BoxZoomTool

# Try to import caiman with proper error handling
try:
    import caiman as cm
    from caiman.source_extraction.cnmf import cnmf, params
except ImportError as e:
    st.error(f"CaImAn import failed: {e}")
    st.info("Please ensure CaImAn is properly installed in your virtual environment.")
    st.stop()

from components.general import plot_calcium_trace_over_time, render_max_proj
from components.parameters import (
    caiman_settings_cnmf_and_plot,
    caiman_settings_component_evaluation,
)
from components.scipy_fix import apply_scipy_fix, remove_scipy_fix
from components.utils import (
    BOKEH_FIGURES,
    custom_show,
    download_calcium_traces,
    download_cnmf_result,
    download_roi_file,
    load_lif_and_tif,
    save_cnmf_result,
    st_load_memmap,
    st_save_memmap,
    use_file_for_bokeh,
    view_components_bokeh,
    view_quilt_bokeh,
)

bpl.output_notebook()
hv.extension("bokeh")

st.set_page_config(
    layout="wide",
    page_title="CNMF ST Analysis",
    page_icon=":microscope:",
    initial_sidebar_state="expanded",
)

# monkey-patch bokeh.plotting.show
bpl.show = custom_show
st.bokeh_chart = use_file_for_bokeh

DATA_DIR = Path().resolve() / "data"


def initialize_multiprocessing_cluster():
    """Initialize or retrieve a multiprocessing cluster for parallel processing."""
    if "cluster" not in st.session_state:
        if active_clusters := multiprocessing.active_children():
            for cluster in active_clusters:
                cluster.terminate()

        try:
            # Stop any existing cluster first
            cm.stop_server()
            st.info("Stopped any existing CaImAn servers")
        except:
            pass
        
        try:
            _, cluster, n_processes = cm.cluster.setup_cluster(
                backend="multiprocessing", n_processes=8, ignore_preexisting=True
            )
            st.session_state["cluster"] = cluster
            st.toast(
                f"Initialized multicore processing with {n_processes} CPU cores.")
        except Exception as e:
            st.error(f"Failed to initialize cluster: {e}")
            # Fallback to single process
            st.session_state["cluster"] = None
            st.warning("Falling back to single-process mode")
    else:
        cluster = st.session_state["cluster"]
        if cluster is not None:
            st.toast(
                f"Using existing multicore processing pool with {cluster._processes} CPU cores."
            )
        else:
            st.info("Using single-process mode")

    return st.session_state["cluster"]


# Initialize multiprocessing cluster
cluster = initialize_multiprocessing_cluster()

# build the page
st.title("CNMF Analysis")
st.write("This is a web app that performs CNMF analysis on calcium imaging data.")

# build left and right columns
project_loading_container = st.container(border=True)
quilt_container = st.container(border=True)

if "CNMFParams" not in st.session_state:
    st.session_state["CNMFParams"] = params.CNMFParams()

# Container for project loading
with project_loading_container:
    col1, col2 = st.columns(2)
    with col1:
        # Check if DATA_DIR exists, if not create it or show error
        if not DATA_DIR.exists():
            st.error(f"Data directory not found: {DATA_DIR}")
            st.info("Please create a 'data' directory and place your .lif files there, or update the DATA_DIR path in the code.")
            st.stop()
        
        # Get list of .lif files
        lif_files = list(DATA_DIR.rglob("*.lif"))
        if not lif_files:
            st.error(f"No .lif files found in {DATA_DIR}")
            st.info("Please place some .lif files in the data directory.")
            st.stop()
        
        # Select a .lif file from the DATA_DIR
        lif_filepath = Path(
            st.selectbox(
                "Select a .lif file",
                lif_files,
                format_func=lambda path: f"{path.parent.name}/{path.name}",
            )
        ).resolve()

        # Look for .tif files that match the .lif filename pattern
        # Files are named like: "slice5.lif - slice5_Merging001.tif"
        lif_name = lif_filepath.name  # Include the .lif extension
        tif_pattern = f"{lif_name}*.tif"
        tif_filelist = sorted(lif_filepath.parent.glob(tif_pattern))
        
        if not tif_filelist:
            st.error(f"No .tif files found matching pattern '{tif_pattern}' in {lif_filepath.parent}")
            st.info("Please ensure there are corresponding .tif files in the data directory.")
            # Show available .tif files for debugging
            all_tifs = list(lif_filepath.parent.glob("*.tif"))
            if all_tifs:
                st.info(f"Available .tif files: {[f.name for f in all_tifs]}")
            st.stop()
            
        tif_filepath = Path(
            st.selectbox(
                "Select a .tif file",
                tif_filelist,
                format_func=lambda path: path.name,
            )
        )

        with st.spinner("Loading data..."):
            lif_file, movie_orig, movie_dict = load_lif_and_tif(
                lif_filepath, tif_filepath
            )
            if movie_dict["number"] is None:
                st.write(
                    f"Series **{movie_dict['name']}** not found in lif_lsd.image_list."
                )
            else:
                st.write(
                    f"Image number **{movie_dict['number']}** for **{movie_dict['name']}**"
                )

            # hash the tif_filepath and use as base_name for memmap
            tif_filepath_hash = hash(tif_filepath)

            # Use current working directory for caiman data instead of hardcoded paths
            caiman_data_dir = Path.cwd() / "caiman_data"
            temp_dir = caiman_data_dir / "temp"
            model_dir = caiman_data_dir / "model"
            
            # Create directories if they don't exist
            temp_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)

            # Check if models directory exists and copy files if needed
            origin_models = Path.cwd() / "models"
            if origin_models.exists():
                import shutil
                tmp_files = os.listdir(origin_models)
                for file_name in tmp_files:
                    source_file = origin_models / file_name
                    target_file = model_dir / file_name
                    if not target_file.exists():
                        shutil.copy(source_file, target_file)
                        st.info(f"Copied model file: {file_name}")

            # Look for existing memmap files
            memmap_list = [f for f in os.listdir(temp_dir) if f.startswith(str(tif_filepath_hash))]

            if len(memmap_list) > 0:
                mc_memmapped_fname = str(temp_dir / memmap_list[0])
                st.info(f"Found existing memmap file: {memmap_list[0]}")
            else:
                mc_memmapped_fname = st_save_memmap(
                    tif_filepath,
                    base_name=f"{tif_filepath_hash}_memmap_",
                    order="C",
                    border_to_0=0,
                    _dview=cluster,
                )

            st.write(f"Memmap saved to: :blue[{mc_memmapped_fname}]")

            Yr, dims, num_frames = st_load_memmap(mc_memmapped_fname)

            # reshape frames in standard 3d format (T x X x Y)
            images = np.reshape(Yr.T, [num_frames] + list(dims), order="F")
            if images.filename is None:
                images.filename = mc_memmapped_fname
            
            # Ensure data is in the correct format for CNMF
            if images.dtype != np.float32:
                st.info(f"Converting data from {images.dtype} to float32")
                # For memmap, we need to be careful about memory usage
                if hasattr(images, 'filename'):
                    # Data is memory mapped, conversion might be memory intensive
                    st.warning("Data type conversion for memory mapped arrays - this may use significant memory")
                images = images.astype(np.float32)
            
            # Validate data before CNMF
            st.info(f"Data validation:")
            st.info(f"- Image dimensions: {images.shape}")
            st.info(f"- Data type: {images.dtype}")
            st.info(f"- Data range: [{np.min(images):.2f}, {np.max(images):.2f}]")
            st.info(f"- Data std: {np.std(images):.2f}")
            st.info(f"- Memory mapped: {hasattr(images, 'filename') and images.filename is not None}")
            st.info(f"- Contains NaN/Inf: {np.any(np.isnan(images)) or np.any(np.isinf(images))}")
            st.info(f"- Array is Fortran ordered: {np.isfortran(images)}")
            
            # Check a small sample of the data
            sample_frame = images[0]  # First frame
            st.info(f"- Sample frame range: [{np.min(sample_frame):.2f}, {np.max(sample_frame):.2f}]")
            st.info(f"- Sample frame mean: {np.mean(sample_frame):.2f}")
            
            # Check if data is reasonable for calcium imaging
            if np.min(images) < 0:
                st.warning("Data contains negative values - this is unusual for calcium imaging")
            if np.max(images) == np.min(images):
                st.error("Data has no variation - all pixels have the same value")
                st.stop()
            if np.any(np.isnan(images)) or np.any(np.isinf(images)):
                st.error("Data contains NaN or infinite values")
                st.stop()
            if np.std(images) < 1.0:
                st.warning(f"Data has very low variation (std={np.std(images):.3f}) - CNMF may struggle to find components")
    with col2:
        correlation_image_orig, fig = render_max_proj(movie_orig)
        st.session_state["correlation_image_orig"] = correlation_image_orig
        st.pyplot(fig, use_container_width=True)

caiman_settings_cnmf_and_plot()


with st.expander("Show CNMF Parameters"):
    st.json(st.session_state.get("CNMFParams").to_json())


cnmf_container = st.container(border=True)
component_evaluation_container = st.container(border=True)
with cnmf_container:
    cnmf_fit_col, cnmf_refit_col = st.columns(2)

    with cnmf_fit_col:
        if st.button("Run CNMF"):
            with st.spinner("Running CNMF..."):
                try:
                    # Check prerequisites
                    cluster = st.session_state["cluster"]
                    if cluster is not None:
                        st.info(f"Cluster processes: {cluster._processes}")
                        n_processes = cluster._processes
                    else:
                        st.info("Using single-process mode")
                        n_processes = 1
                    
                    st.info(f"Images shape: {images.shape}")
                    st.info(f"Images type: {type(images)}")
                    st.info(f"Images filename: {getattr(images, 'filename', 'No filename')}")
                    
                    # Verify parameters
                    params = st.session_state.get("CNMFParams")
                    if params is None:
                        st.error("CNMFParams is None")
                        st.stop()
                    
                    st.info(f"CNMF Parameters initialized successfully")
                    
                    # Create CNMF model
                    cnmf_model = cnmf.CNMF(
                        n_processes=n_processes,
                        params=params,
                        dview=cluster,
                    )
                    st.info("CNMF model created successfully")
                    
                    # Verify the model was created properly
                    if cnmf_model is None:
                        st.error("CNMF model creation failed")
                        st.stop()
                    
                    # Debug the model attributes
                    st.info(f"CNMF model type: {type(cnmf_model)}")
                    st.info(f"CNMF model params: {cnmf_model.params is not None}")
                    st.info(f"CNMF model dview: {cnmf_model.dview}")
                    st.info(f"CNMF model estimates: {cnmf_model.estimates}")
                    
                    # Check if this is a patched vs non-patched run
                    rf_value = params.get('patch', 'rf')
                    st.info(f"rf parameter value: {rf_value}")
                    if rf_value is None:
                        st.info("Running in NON-PATCHED mode (rf=None)")
                    else:
                        st.info(f"Running in PATCHED mode (rf={rf_value})")
                    
                    # Run the fit
                    st.info("Starting CNMF fitting...")
                    
                    # Log the important parameters being used
                    st.info(f"Key parameters: rf={params.get('patch', 'rf')}, K={params.get('init', 'K')}, only_init={params.get('patch', 'only_init')}")
                    st.info(f"Data parameters: p={params.get('preprocess', 'p')}, nb={params.get('init', 'nb')}")
                    st.info(f"Cluster: {cluster}, n_processes: {n_processes}")
                    
                    # Capture any stdout/stderr during fitting
                    import sys
                    from io import StringIO
                    
                    # Create a StringIO object to capture output
                    captured_output = StringIO()
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    
                    try:
                        # Redirect stdout and stderr to capture any print statements or errors
                        sys.stdout = captured_output
                        sys.stderr = captured_output
                        
                        # Apply SciPy compatibility fix before fitting
                        st.info("Applying SciPy compatibility fix for CaImAn deconvolution...")
                        apply_scipy_fix()
                        
                        # Try to call fit method and catch any issues
                        st.info("About to call cnmf_model.fit(images)...")
                        
                        # Wrap the fit call in a try-catch to catch any exceptions that might be swallowed
                        try:
                            cnmf_fit = cnmf_model.fit(images)
                            st.info(f"cnmf_model.fit() returned: {cnmf_fit}")
                            st.info(f"cnmf_fit type: {type(cnmf_fit)}")
                            st.info(f"cnmf_fit is cnmf_model: {cnmf_fit is cnmf_model}")
                        except Exception as inner_e:
                            st.error(f"Exception caught during fit(): {inner_e}")
                            st.error(f"Exception type: {type(inner_e).__name__}")
                            import traceback
                            st.error(f"Inner exception traceback: {traceback.format_exc()}")
                            cnmf_fit = None
                        
                        # Additional debugging
                        if cnmf_fit is not None:
                            st.info("cnmf_fit is not None - checking attributes...")
                            if hasattr(cnmf_fit, 'estimates'):
                                st.info(f"cnmf_fit.estimates: {cnmf_fit.estimates}")
                                if cnmf_fit.estimates is not None:
                                    st.info(f"cnmf_fit.estimates type: {type(cnmf_fit.estimates)}")
                                    if hasattr(cnmf_fit.estimates, 'A'):
                                        st.info(f"cnmf_fit.estimates.A: {cnmf_fit.estimates.A}")
                                        if cnmf_fit.estimates.A is not None:
                                            st.info(f"A matrix shape: {cnmf_fit.estimates.A.shape}")
                        else:
                            st.error("cnmf_fit is None after fit() call")
                            # Check if cnmf_model itself has estimates populated despite None return
                            st.info("Checking if cnmf_model was modified in place...")
                            if hasattr(cnmf_model, 'estimates') and cnmf_model.estimates is not None:
                                st.info("cnmf_model.estimates was populated despite None return!")
                                if hasattr(cnmf_model.estimates, 'A') and cnmf_model.estimates.A is not None:
                                    st.info(f"cnmf_model.estimates.A shape: {cnmf_model.estimates.A.shape}")
                                    st.warning("Using cnmf_model as the result since it was modified in place")
                                    cnmf_fit = cnmf_model
                        
                    finally:
                        # Restore original stdout and stderr
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                        
                        # Remove SciPy compatibility fix
                        remove_scipy_fix()
                        
                        # Get the captured output
                        output = captured_output.getvalue()
                        if output:
                            st.text_area("CNMF Output Log:", output, height=200)
                    
                    # Check result
                    if cnmf_fit is None:
                        st.error("CNMF fitting failed and returned None.")
                        st.error("This could be due to:")
                        st.error("- Invalid parameters")
                        st.error("- Corrupted data") 
                        st.error("- Memory issues")
                        st.error("- Cluster communication problems")
                        st.error("- Insufficient components found during initialization")
                        
                        # Check if cnmf_model itself has been modified during fit
                        st.info("Checking cnmf_model state after failed fit...")
                        st.info(f"cnmf_model type: {type(cnmf_model)}")
                        st.info(f"cnmf_model estimates: {cnmf_model.estimates}")
                        if hasattr(cnmf_model, 'estimates') and cnmf_model.estimates is not None:
                            st.info("cnmf_model.estimates exists - checking if it was populated...")
                            if hasattr(cnmf_model.estimates, 'A'):
                                st.info(f"cnmf_model.estimates.A: {cnmf_model.estimates.A}")
                        
                        # Use cnmf_model instead since fit() should have modified it in place
                        st.warning("Trying to use cnmf_model as the result instead of None return value...")
                        cnmf_fit = cnmf_model
                        
                    else:
                        st.success("CNMF fitting completed successfully!")
                        st.info(f"CNMF fit type: {type(cnmf_fit)}")
                        if hasattr(cnmf_fit, 'estimates'):
                            st.info(f"Estimates available: {cnmf_fit.estimates is not None}")
                            if cnmf_fit.estimates is not None:
                                if hasattr(cnmf_fit.estimates, 'A') and cnmf_fit.estimates.A is not None:
                                    st.info(f"Components shape: {cnmf_fit.estimates.A.shape}")
                                    st.info(f"Number of components found: {cnmf_fit.estimates.A.shape[1]}")
                                else:
                                    st.warning("No spatial components (A matrix) found")
                                if hasattr(cnmf_fit.estimates, 'C') and cnmf_fit.estimates.C is not None:
                                    st.info(f"Temporal components shape: {cnmf_fit.estimates.C.shape}")
                                else:
                                    st.warning("No temporal components (C matrix) found")
                    
                    st.session_state["cnmf_fit"] = cnmf_fit
                        
                except Exception as e:
                    st.error(f"Error during CNMF fitting: {e}")
                    st.error(f"Error type: {type(e).__name__}")
                    import traceback
                    st.error(f"Full traceback: {traceback.format_exc()}")
                    st.session_state["cnmf_fit"] = None

        if "cnmf_fit" in st.session_state and st.session_state["cnmf_fit"] is not None:
            cnmf_fit = st.session_state["cnmf_fit"]
            if cnmf_fit.estimates is not None:
                cnmf_fit.estimates.plot_contours_nb(img=correlation_image_orig)
                st.bokeh_chart(BOKEH_FIGURES[-1], use_container_width=True)
            else:
                st.warning("CNMF fit completed but no estimates found.")

    with cnmf_refit_col:
        if "cnmf_fit" in st.session_state and st.session_state["cnmf_fit"] is not None:
            cnmf_fit = st.session_state["cnmf_fit"]
            if st.button("Refit CNMF"):
                with st.spinner("Refitting CNMF..."):
                    cnmf_refit = cnmf_fit.refit(images)
                    st.session_state["cnmf_refit"] = cnmf_refit

        if "cnmf_refit" in st.session_state and st.session_state["cnmf_refit"] is not None:
            cnmf_refit = st.session_state["cnmf_refit"]
            if cnmf_refit.estimates is not None:
                cnmf_refit.estimates.plot_contours_nb(img=correlation_image_orig)
                st.bokeh_chart(BOKEH_FIGURES[-1])
            else:
                st.warning("CNMF refit completed but no estimates found.")

    # put all cnmf_*fit objects from the session state into a list
    pattern = re.compile(r"^cnmf_.*fit$")
    cnmf_options = [
        key
        for key, value in sorted(st.session_state.items())
        if pattern.match(key) and value is not None
    ]
    
    if cnmf_options:
        selected_key = st.selectbox(
            label="Select a CNMF fit to work with",
            options=cnmf_options,
            format_func=lambda k: " ".join(k.lower().split("_")[:]),
        )
        cnmf_selected = st.session_state["cnmf_selected"] = st.session_state.get(selected_key, None)
    else:
        cnmf_selected = None
        st.info("No CNMF fits available. Please run CNMF first.")

    if cnmf_selected and cnmf_selected.estimates:
        st.write(f"Estimates shape: {cnmf_selected.estimates.A.shape}")
        st.write(f"Components shape: {cnmf_selected.estimates.C.shape}")
        st.write(f"Residuals shape: {cnmf_selected.estimates.YrA.shape}")
        st.write(f"Spikes shape: {cnmf_selected.estimates.S.shape}")

with component_evaluation_container:
    caiman_settings_component_evaluation()

    if st.button("Evaluate Components") or "cnmf_evaluated" in st.session_state:
        if cnmf_selected is None:
            st.error("Please select a CNMF fit to evaluate components.")
            st.stop()
        with st.spinner("Evaluating Components..."):
            cnmf_selected = st.session_state["cnmf_selected"]
            cluster = st.session_state.get("cluster")
            st.session_state["cnmf_evaluated"] = cnmf_evaluated = (
                cnmf_selected.estimates.evaluate_components(
                    images, cnmf_selected.params, dview=cluster,
                )
            )

            st.write(
                f"Num accepted/rejected: {len(cnmf_selected.estimates.idx_components)}, {len(cnmf_selected.estimates.idx_components_bad)}"
            )

        if "cnmf_evaluated" in st.session_state:
            cnmf_evaluated = st.session_state["cnmf_evaluated"]
            cnmf_evaluated.plot_contours_nb(
                img=correlation_image_orig, idx=cnmf_evaluated.idx_components
            )

            figures_row = row(BOKEH_FIGURES[-2], BOKEH_FIGURES[-1])
            st.bokeh_chart(figures_row)

            if cnmf_evaluated.F_dff is None:
                with st.spinner('Calculating estimates.F_dff'):
                    cnmf_evaluated.detrend_df_f(quantileMin=8,
                                                frames_window=250,
                                                flag_auto=False,
                                                use_residuals=False)
            else:
                st.toast("Estimates.F_dff already calculated")

            col1, col2 = st.columns(2)

            with col1:
                st.write("### Accepted Components")
                if len(cnmf_evaluated.idx_components) > 0:
                    cnmf_evaluated.nb_view_components(
                        img=correlation_image_orig,
                        idx=cnmf_evaluated.idx_components,
                        cmap="gray",
                        denoised_color="green",
                    )

                    view_components_bokeh()

                    download_roi_file(
                        cnmf_evaluated, zip_file_name=f"ROIS_good_{tif_filepath.stem}")

                    # st.write(cnmf_evaluated)

                    download_cnmf_result(cnmf_selected, correlation_image_orig=correlation_image_orig,
                                         zip_file_name=f"Results_{tif_filepath.with_suffix('.hdf5').name}")

                    save_cnmf_result(cnmf_selected, correlation_image_orig=correlation_image_orig,
                                     save_file_name=f"Results_{tif_filepath.with_suffix('.hdf5').name}")

                    download_calcium_traces(
                        cnmf_evaluated, zip_file_name=f"calcium_traces_{tif_filepath.stem}.zip")

                else:
                    st.info("No components were accepted.")

                plot_calcium_trace_over_time(num_frames, cnmf_selected, cnmf_evaluated)

            with col2:
                st.write("### Rejected Components")
                if len(cnmf_evaluated.idx_components_bad) > 0:
                    cnmf_evaluated.nb_view_components(
                        img=correlation_image_orig,
                        idx=cnmf_evaluated.idx_components_bad,
                        cmap="gray",
                        denoised_color="red",
                    )

                    view_components_bokeh()

                    download_roi_file(
                        cnmf_evaluated, zip_file_name=f"ROIS_bad_{tif_filepath.stem}")

                    # st.write(cnmf_selected)

                else:
                    st.info("No components were rejected.")

    if st.button("Run Simple CNMF Test (Skip Deconvolution)"):
        with st.spinner("Running Simple CNMF Test without deconvolution..."):
            try:
                st.info("Testing CNMF with deconvolution disabled...")
                
                # Create parameters that skip deconvolution to avoid SciPy issues
                test_params = params.CNMFParams()
                test_params.set('data', {'fr': 28.0})
                test_params.set('preprocess', {'p': 0})  # No deconvolution (p=0)
                test_params.set('init', {
                    'nb': 1,
                    'K': 5,  # More components
                    'gSig': [4, 4],  # Slightly larger
                    'method_init': 'greedy_roi',
                    'ssub': 1,  # No downsampling
                    'tsub': 1,
                })
                test_params.set('patch', {
                    'rf': None,  # No patches - process entire FOV
                    'only_init': False,
                })
                test_params.set('merging', {'merge_thr': 0.85})
                test_params.set('temporal', {'p': 0})  # Explicitly disable deconvolution
                
                # Create simple CNMF model (no patches, single process)
                test_cnmf = cnmf.CNMF(
                    n_processes=1,
                    params=test_params,
                    dview=None,  # No cluster
                )
                
                st.info("Created test CNMF model without deconvolution, running fit...")
                
                # Apply SciPy compatibility fix before fitting (even though deconv is disabled)
                apply_scipy_fix()
                
                test_fit = test_cnmf.fit(images)
                
                if test_fit is None:
                    st.error("CNMF test without deconvolution also failed")
                    # Check if test_cnmf was modified in place
                    if hasattr(test_cnmf, 'estimates') and test_cnmf.estimates is not None:
                        if hasattr(test_cnmf.estimates, 'A') and test_cnmf.estimates.A is not None:
                            st.success("Found components in test_cnmf despite None return!")
                            st.info(f"Components found: {test_cnmf.estimates.A.shape[1]}")
                            st.session_state["cnmf_test_no_deconv"] = test_cnmf
                        else:
                            st.error("No estimates found in test_cnmf either")
                    else:
                        st.error("test_cnmf.estimates is None")
                else:
                    st.success("CNMF test without deconvolution succeeded!")
                    st.info(f"Test found {test_fit.estimates.A.shape[1] if test_fit.estimates.A is not None else 0} components")
                    st.session_state["cnmf_test_no_deconv"] = test_fit
                    
            except Exception as e:
                st.error(f"CNMF test without deconvolution failed: {e}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
            finally:
                # Remove SciPy compatibility fix
                remove_scipy_fix()

    if st.button("Run Simple CNMF Test"):
            with st.spinner("Running Simple CNMF Test..."):
                try:
                    st.info("Testing CNMF with minimal parameters...")
                    
                    # Create minimal test parameters
                    test_params = params.CNMFParams()
                    test_params.set('data', {'fr': 28.0})
                    test_params.set('preprocess', {'p': 1})  # Simpler deconvolution
                    test_params.set('init', {
                        'nb': 1,
                        'K': 3,  # Very few components
                        'gSig': [3, 3],
                        'method_init': 'greedy_roi',
                        'ssub': 2,
                        'tsub': 2,
                    })
                    test_params.set('patch', {
                        'rf': None,  # No patches - process entire FOV
                        'only_init': False,
                    })
                    test_params.set('merging', {'merge_thr': 0.85})
                    
                    # Create simple CNMF model (no patches, single process)
                    test_cnmf = cnmf.CNMF(
                        n_processes=1,
                        params=test_params,
                        dview=None,  # No cluster
                    )
                    
                    st.info("Created test CNMF model, running fit...")
                    
                    # Apply SciPy compatibility fix before fitting
                    apply_scipy_fix()
                    
                    test_fit = test_cnmf.fit(images)
                    
                    if test_fit is None:
                        st.error("Even simple CNMF test failed - this suggests a fundamental issue")
                    else:
                        st.success("Simple CNMF test succeeded!")
                        st.info(f"Test found {test_fit.estimates.A.shape[1] if test_fit.estimates.A is not None else 0} components")
                        st.session_state["cnmf_test"] = test_fit
                        
                except Exception as e:
                    st.error(f"Simple CNMF test failed: {e}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                finally:
                    # Remove SciPy compatibility fix
                    remove_scipy_fix()
