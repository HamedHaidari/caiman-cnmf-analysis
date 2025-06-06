import streamlit as st
import numpy as np

from components.general import plot_cnmf_patches
from caiman.source_extraction.cnmf import params

from components.utils import view_quilt_bokeh


@st.fragment
def caiman_settings_cnmf_and_plot():
    correlation_image_orig = st.session_state.get("correlation_image_orig")
    quilt_settings_col, quilt_plot_col = st.columns(2)

    with quilt_settings_col:
        # CNMF parameters
        st.write("#### CNMF Parameters")
        col1, col2, col3, col4 = st.columns(4) # number 4 separates the space in 4 equal columns
        rf = col1.number_input("Patch half-size", value=25, help="In pixels. Reduced from 40 to 25 for 512x512 images")
        K = col2.number_input("Components per patch", value=4, help="Reduced from 5 to 4 for better performance")
        stride_cnmf = col3.number_input(
            "CNMF stride", value=20, help="Patch overlap. Reduced from 25 to 20")
        gSig = col4.text_input(
            "Neuron half-width", value="3, 3", help="Expected in pixels. Reduced from 5,5 to 3,3"
        )

        fr_rate = col1.selectbox(
            "Frame rate", [4.0, 28.0], help="Hertz"
        )
        p = col2.number_input("AR order", value=2,
                              help="p: Order of the autoregressive model. p = 0 turns deconvolution off. If transients in your data rise instantaneously, set p = 1 (occurs at low sample rate or slow indicator). If transients have visible rise time, set p = 2. If the wrong order is chosen, spikes are extracted unreliably.")
        gnb = col3.number_input("Global bg components",
                                value=1, help="Reduced to 1 for initial testing. nb: Number of global background components. This is a measure of the complexity of your background noise. Defaults to nb = 2, assuming a relatively homogeneous background. nb = 3 might fit for more complex noise, nb = 1 is usually too low. If nb is set too low, extracted traces appear too noisy, if nb is set too high, neuronal signal starts getting absorbed into the background reduction, resulting in reduced transients.")
        merge_thr = col4.slider(
            "Merge threshold", 0.0, 1.0, 0.98, help="Max correlation"
        )
        
        col1, col2, col3, _ = st.columns([0.25, 0.25, 0.25, 0.25]) 
        # _ is for the forth parameter. Values name the percentage of the available space
        ssub = col1.number_input(
            "Spatial subsample", value=1, help="During init")
        tsub = col2.number_input("Temporal subsample",
                                 value=1, help="During init")
        method_init = col3.selectbox(
            "Init method", ["greedy_roi", "sparse_nmf", "local_NMF"]
        )
        bas_nonneg = st.checkbox(
            "Enforce nonnegativity", value=True, help="On calcium traces"
        )

        # Create a dictionary with all the settings using proper CaImAn parameter groups
        # This fixes the "non-pathed parameters" deprecation warnings
        settings = {
            "data": {
                "fr": fr_rate,
            },
            "preprocess": {
                "p": p,
            },
            "init": {
                "nb": gnb,
                "K": K,
                "gSig": np.array(list(map(int, gSig.split(",")))),
                "gSiz": 2 * np.array(list(map(int, gSig.split(",")))) + 1,
                "method_init": method_init,
                "rolling_sum": True,
                "ssub": ssub,
                "tsub": tsub,
            },
            "patch": {
                "rf": rf,
                "stride": stride_cnmf,
                "only_init": False,  # Set to False to perform full CNMF fitting, not just initialization
            },
            "merging": {
                "merge_thr": merge_thr,
            },
            "temporal": {
                "bas_nonneg": bas_nonneg,
            },
        }

        assert "CNMFParams" in st.session_state, "CNMFParams not in session state"
        cnmf_params = st.session_state.get("CNMFParams")
        
        # Set parameters using the proper grouped structure to avoid deprecation warnings
        cnmf_params.set('data', {'fr': fr_rate})
        cnmf_params.set('preprocess', {'p': p})
        cnmf_params.set('init', {
            'nb': gnb,
            'K': K,
            'gSig': np.array(list(map(int, gSig.split(",")))),
            'gSiz': 2 * np.array(list(map(int, gSig.split(",")))) + 1,
            'method_init': method_init,
            'rolling_sum': True,
            'ssub': ssub,
            'tsub': tsub,
        })
        cnmf_params.set('patch', {
            'rf': rf,
            'stride': stride_cnmf,
            'only_init': False,  # Set to False to perform full CNMF fitting
        })
        cnmf_params.set('merging', {'merge_thr': merge_thr})
        cnmf_params.set('temporal', {'bas_nonneg': bas_nonneg})
        
        st.info("Parameters updated successfully using proper CaImAn parameter groups")
    with quilt_plot_col:
        fig, ax = plot_cnmf_patches(
            st.session_state.get("CNMFParams"), correlation_image_orig)
        plot_cnmf_patches(
            st.session_state.get("CNMFParams"), correlation_image_orig)

        # view_quilt_bokeh()
        st.pyplot(fig, use_container_width=False)

    return


def caiman_settings_component_evaluation():
    st.write("#### Component Evaluation Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        rval_lowest = st.number_input(
            "rval_lowest", value=-1.0, min_value=-1.0, step=0.1, help="Reject below this")
        rval_thr = st.slider("rval_thr", 0.0, 1.0, 0.85,
                             help="For accepting component")

    with col2:
        SNR_lowest = st.number_input(
            "SNR_lowest", value=0.5, min_value=0.0, step=0.1, help="Reject below this")
        min_SNR = st.slider("min_SNR", 0.0, 20.0,
                            5.0, help="For accepting component")

    with col3:
        cnn_lowest = st.number_input(
            "cnn_lowst", value=0.1, min_value=0.0, max_value=1.0, step=0.1, help="Reject below this")
        min_cnn_thr = st.slider("min_cnn_thr", 0.0, 1.0,
                                0.99, help="CNN classifier threshold")

    settings = {
        "rval_lowest": rval_lowest,
        "rval_thr": rval_thr,
        "SNR_lowest": SNR_lowest,
        "min_SNR": min_SNR,
        "cnn_lowest": cnn_lowest,
        "min_cnn_thr": min_cnn_thr,
    }

    assert "CNMFParams" in st.session_state, "CNMFParams not in session state"
    st.session_state.get("CNMFParams").change_params(settings)

    return


# fmt: off
def caiman_settings(lif_lsd, IMAGE_NUM):
    # Retrieve values from the movie
    fr_from_movie = 1 / \
        float(lif_lsd.get_image(IMAGE_NUM).info["settings"]["FrameTime"])
    dxy_from_movie = lif_lsd.get_image(IMAGE_NUM).info["scale"][:2]

    st.write("## CNMF Settings")

    # General dataset-dependent parameters
    st.write("#### General Parameters")
    col1, col2, col3 = st.columns(3)
    fr = col1.number_input("Imaging rate (fps)", value=fr_from_movie, help="Frames per second")
    decay_time = col2.number_input("Decay time (s)", value=0.8, help="Typical transient length")
    dxy = col3.text_input("Spatial resolution (um/px)", value=str(dxy_from_movie), help="Resolution in x and y")

    # Motion correction parameters
    # st.write("#### Motion Correction Parameters")
    # col1, col2, col3, col4 = st.columns(4)
    # strides = col1.text_input("Strides", value="48, 48", help="New patch every x pixels")
    # max_shifts = col2.text_input("Max shifts", value="6, 6", help="Max allowed rigid shifts")
    # overlaps = col3.text_input("Overlaps", value="24, 24", help="Overlap between patches")
    # max_deviation_rigid = col4.number_input("Max deviation rigid", value=3, help="Max shifts deviation")
    # pw_rigid = st.checkbox("PW Rigid", value=True, help="Non-rigid motion correction")

    # CNMF parameters
    st.write("#### CNMF Parameters")
    col1, col2, col3, col4 = st.columns(4)
    rf = col1.number_input("Patch half-size", value=40, help="In pixels")
    K = col2.number_input("Components per patch", value=5)
    stride_cnmf = col3.number_input("CNMF stride", value=25, help="Patch overlap")
    gSig = col4.text_input("Neuron half-width", value="5, 5", help="Expected in pixels")

    fr_rate = col1.selectbox("Frame rate", [4.0, 28.0], help="Frames per second (fps)")
    p = col2.number_input("AR order", value=2, help="Autoregressive system order")
    gnb = col3.number_input("Global bg components", value=1, help="Set to 1 or 2")
    merge_thr = col4.slider("Merge threshold", 0.0, 1.0, 0.85, help="Max correlation")
    col1, col2, col3, _ = st.columns([0.25, 0.25, 0.25, 0.25])
    method_init = col1.selectbox("Init method", ['greedy_roi', 'sparse_nmf', 'local_NMF'])
    ssub = col2.number_input("Spatial subsample", value=1, help="During init")
    tsub = col3.number_input("Temporal subsample", value=2, help="During init")
    bas_nonneg = st.checkbox("Enforce nonnegativity", value=True, help="On calcium traces")

    # Component evaluation parameters
    # st.write("#### Component Evaluation Parameters")
    # col1, col2, col3, col4 = st.columns(4)
    # min_SNR = col1.number_input("Min SNR", value=2.0, help="For accepting component")
    # cnn_thr = col2.slider("CNN threshold", 0.0, 1.0, 0.99, help="CNN classifier threshold")
    # rval_thr = col3.slider("Space corr threshold", 0.0, 1.0, 0.85, help="For accepting component")
    # cnn_lowest = col4.slider("CNN lowest prob", 0.0, 1.0, 0.1, help="Reject below this")

    # strip the string of all characters except digits and dots
    dxy = "".join(filter(lambda x: x.isdigit() or x == ",", dxy))

    # Create a dictionary with all the settings
    settings = {
        "fr": fr_rate,
        "decay_time": decay_time,
        "dxy": tuple(map(float, dxy.split(","))),
        # "strides": tuple(map(int, strides.split(","))),
        # "overlaps": tuple(map(int, overlaps.split(","))),
        # "max_shifts": tuple(map(int, max_shifts.split(","))),
        # "max_deviation_rigid": max_deviation_rigid,
        # "pw_rigid": pw_rigid,
        "p": p,
        "nb": gnb,
        "rf": rf,
        "K": K,
        "gSig": np.array(list(map(int, gSig.split(",")))),
        "gSiz": 2 * np.array(list(map(int, gSig.split(",")))) + 1,
        "stride": stride_cnmf,
        "method_init": method_init,
        "rolling_sum": True,
        "only_init": True,
        "ssub": ssub,
        "tsub": tsub,
        "merge_thr": merge_thr,
        "bas_nonneg": bas_nonneg,
        # "min_SNR": min_SNR,
        # "rval_thr": rval_thr,
        # "use_cnn": True,
        # "min_cnn_thr": cnn_thr,
        # "cnn_lowest": cnn_lowest,
    }

    with st.expander("Show settings"):
        st.json(settings)

    return settings

# fmt: on
