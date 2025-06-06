import numpy as np
import caiman as cm
import matplotlib.pyplot as plt
import streamlit as st
from caiman.utils.visualization import view_quilt
from caiman.source_extraction.cnmf import cnmf, params


@st.cache_data()
def render_max_proj(movie_orig):
    max_projection_orig = np.max(movie_orig, axis=0)
    correlation_image_orig = cm.local_correlations(movie_orig, swap_dim=False)
    # get rid of NaNs, if they exist
    correlation_image_orig[np.isnan(correlation_image_orig)] = 0

    f, (ax_max, ax_corr) = plt.subplots(1, 2, figsize=(4, 4))
    ax_max.imshow(
        max_projection_orig,
        cmap="viridis",
        vmin=np.percentile(np.ravel(max_projection_orig), 50),
        vmax=np.percentile(np.ravel(max_projection_orig), 99.5),
    )
    ax_max.set_title("Max Projection Orig", fontsize=8)

    ax_corr.imshow(
        correlation_image_orig,
        cmap="viridis",
        vmin=np.percentile(np.ravel(correlation_image_orig), 50),
        vmax=np.percentile(np.ravel(correlation_image_orig), 99.5),
    )
    ax_corr.set_title("Correlation Image Orig", fontsize=8)
    return correlation_image_orig, f


def plot_cnmf_patches(parameters, correlation_image_orig):
    # Calculate stride and overlap from parameters
    cnmf_patch_width = parameters.patch["rf"] * 2 + 1
    cnmf_patch_overlap = parameters.patch["stride"] + 1
    cnmf_patch_stride = cnmf_patch_width - cnmf_patch_overlap

    # Display calculated values
    st.write(
        f"Patch width: {cnmf_patch_width}, Stride: {cnmf_patch_stride}, Overlap: {cnmf_patch_overlap}"
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(1.5, 1.5))

    # Set tick labels font size
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.tick_params(axis="both", which="minor", labelsize=6)

    patch_ax = view_quilt(
        correlation_image_orig,
        cnmf_patch_stride,
        cnmf_patch_overlap,
        vmin=np.percentile(np.ravel(correlation_image_orig), 50),
        vmax=np.percentile(np.ravel(correlation_image_orig), 99.5),
        ax=ax,
    )

    # from caiman.utils.visualization import nb_view_quilt

    # nb_view_quilt(correlation_image_orig,
    #               rf=parameters.patch["rf"], stride_input=parameters.patch["stride"])

    return fig, patch_ax


def plot_calcium_trace_over_time(num_frames, cnmf_selected, cnmf_evaluated):
    #frame_rate = cnmf_selected.params.data['fr']
    #frame_rate = cnmf_selected.parameters.data['fr']
    frame_rate = 27.637
    frame_pd = 1/frame_rate
    frame_times = np.linspace(0, num_frames*frame_pd, num_frames) 
    # Beispiel: num_frames = insgesamt 4800 frames. frame_rate = 10 frames pro Sekunde. -> 4800/10 = 480 Sekunden = 8 Minuten

    import matplotlib.pyplot as plt
    # plot F_dff
    component_number = len(cnmf_evaluated.idx_components)
    f, ax = plt.subplots(figsize=(4, 2))
    ax.plot(frame_times,
            cnmf_evaluated.F_dff[component_number, :],
            linewidth=0.5,
            color='k')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$\Delta F/F$')
    ax.set_title(f"$\Delta F/F$ for ROI {len(cnmf_evaluated.idx_components)}")
    # set top and right spine invisible
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # set y limit between 0 and 1
    #ax.set_xlim([-1, max(frame_times)])
    ax.set_xticks(np.arange(0, max(frame_times), 20))
    ax.set_ylim([-0.05, max(cnmf_evaluated.F_dff[component_number, :])])
    #ax.set_yticks(np.arange(0, max(cnmf_evaluated.F_dff[component_number, :]), 0.1))
    #plt.grid(True)
    #plt.rc('axes', labelsize=10)
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)

    st.pyplot(f, use_container_width=True)
    return
