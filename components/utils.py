import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import caiman as cm
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from bokeh.io import output_file
from bokeh.layouts import Column, Row
from bokeh.models import Slider, BoxZoomTool, PanTool
from bokeh.plotting import figure, save
from readlif.reader import LifFile
from roifile import ImagejRoi
import tempfile


# Store figures in a global list
BOKEH_FIGURES = []


def collect_figures(layout):
    """Recursively collect figures and sliders from a Bokeh layout."""
    figures = []
    if isinstance(layout, (figure, Slider)):
        figures.append(layout)
    elif isinstance(layout, (Row, Column)):
        for child in layout.children:
            figures.extend(collect_figures(child))
    return figures


def custom_show(*args, **kwargs):
    """Define custom show function"""
    global BOKEH_FIGURES
    for arg in args:
        BOKEH_FIGURES.extend(collect_figures(arg))
        if not isinstance(arg, (figure, Row, Column, Slider)):
            print(f"Unhandled type: {type(arg)}")


# def use_file_for_bokeh(chart: figure, chart_height=500, chart_width=500, use_container_width=False):
#     output_file("bokeh_graph.html")
#     save(chart)
#     with open("bokeh_graph.html", "r", encoding="utf-8") as f:
#         html = f.read()
#     components.html(html, height=chart_height)

def use_file_for_bokeh(chart: figure, chart_height=500, chart_width=500, use_container_width=False):
    output_file("bokeh_graph.html")
    save(chart)
    with open("bokeh_graph.html", "r", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=chart_height)


def download_roi_file(cnmf_evaluated, zip_file_name=None):
    roi_dir_name = "roi_files"
    if zip_file_name is None:
        zip_file_name = "roi_files.zip"

    roi_dir_path = Path.cwd() / roi_dir_name
    if roi_dir_path.exists():
        shutil.rmtree(roi_dir_path)
    roi_dir_path.mkdir(exist_ok=False)

    idx_accepted = cnmf_evaluated.idx_components
    all_contour_coords = [cnmf_evaluated.coordinates[idx]
                          ["coordinates"] for idx in idx_accepted]

    # write coordinates where there is more than 4 NANs

    for index, coords in enumerate(all_contour_coords):
        roi = ImagejRoi.frompoints(coords[1:-1, :])
        roi.tofile(os.path.join(roi_dir_path, f"caiman-{index+1:04d}.roi"))

    # Zip the directory
    zip_file_path = shutil.make_archive(roi_dir_name, 'zip', roi_dir_path)

    # Provide the download button for the zip file
    with open(zip_file_path, "rb") as file:
        btn = st.download_button(
            label="Download ROI Files",
            data=file,
            file_name=zip_file_name,
            mime="application/zip",
        )

    if btn:
        st.success("ROI files downloaded successfully!")


def download_cnmf_result(cnmf_object, correlation_image_orig, zip_file_name=None):
    if zip_file_name is None:
        zip_file_name = "cnmf_result.hdf5"

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / zip_file_name

        cnmf_object.estimates.Cn = correlation_image_orig
        cnmf_object.save(filepath.as_posix())

        with open(filepath, "rb") as file:
            st.download_button(
                label="Download CNMF Result",
                data=file,
                file_name=zip_file_name,
                mime="application/hdf5",
            )


def save_cnmf_result(cnmf_object, correlation_image_orig, save_file_name=None):
    if save_file_name is None:
        save_file_name = "cnmf_result.hdf5"

    os.makedirs("./results", exist_ok=True)

    filepath = Path("./results") / save_file_name

    if st.button("Save CNMF Result"):
        cnmf_object.estimates.Cn = correlation_image_orig
        cnmf_object.save(filepath.as_posix())


def download_calcium_traces(cnmf_evaluated, zip_file_name=None, accepted_only=True):
    if zip_file_name is None:
        zip_file_name = "calcium_traces.zip"

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "traces"
        filepath.mkdir(parents=True, exist_ok=True)

        for index in cnmf_evaluated.idx_components:
            np.save(
                filepath / f"caiman_accepted_fdf-{index+1:04d}.npy", cnmf_evaluated.F_dff[index, :])
            np.save(
                filepath / f"caiman_accepted_raw-{index+1:04d}.npy", cnmf_evaluated.YrA[index, :])


        if not accepted_only:
            for index in cnmf_evaluated.idx_components_bad:
                np.save(
                    filepath / f"caiman_rejected_fdf-{index+1:04d}.npy", cnmf_evaluated.F_dff[index, :])
                np.save(
                    filepath / f"caiman_rejected_raw-{index+1:04d}.npy", cnmf_evaluated.YrA[index, :])

        zip_file_path = shutil.make_archive(filepath, 'zip', filepath)

        with open(zip_file_path, "rb") as file:
            st.download_button(
                label="Download Calcium Traces",
                data=file,
                file_name=zip_file_name,
                mime="application/zip",
            )

        return


@ st.cache_resource()
def load_lif_and_tif(
    lif_filepath: Path, tif_filepath: Path
) -> Tuple[LifFile, LifFile, cm.movie, Dict[str, Any]]:
    """Load the LIF and TIF files"""
    lif_file = LifFile(lif_filepath)
    tif_file = cm.load(tif_filepath)

    # Extract the image name from the TIF filename
    # Files can use either " - " or "__" as separators
    tif_stem = tif_filepath.stem
    if " - " in tif_stem:
        # Format: "filename.lif - description.tif"
        tif_name = tif_stem.split(" - ")[1]
    elif "__" in tif_stem:
        # Format: "filename__description.tif"
        tif_name = tif_stem.split("__")[1]
    else:
        # Fallback: use the entire stem
        tif_name = tif_stem
    
    tif_number = next(
        (
            index
            for index, image_dict in enumerate(lif_file.image_list)
            if image_dict.get("name") == tif_name
        ),
        None,
    )

    return lif_file, tif_file, {"name": tif_name, "number": tif_number}


@ st.cache_resource()
def st_save_memmap(
    movie_path: Path,
    base_name: str = "memmap_",
    order: str = "C",
    border_to_0: int = 0,
    _dview: Any = None,
) -> str:
    """Save the memmap"""
    mc_memmapped_fname = cm.save_memmap(
        [movie_path.as_posix()],
        base_name=base_name,
        order=order,
        border_to_0=border_to_0,
        dview=_dview,
    )
    return mc_memmapped_fname


@ st.cache_resource()
def st_load_memmap(mc_memmapped_fname: str) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """Load the memmap"""
    Yr, dims, num_frames = cm.load_memmap(mc_memmapped_fname)
    images = np.reshape(Yr.T, [num_frames] + list(dims), order="F")
    return images, dims, num_frames


def view_components_bokeh():
    figures_dict: dict = {
        "Slider": BOKEH_FIGURES[-4],
        "Components": BOKEH_FIGURES[-3],
        "Stats": BOKEH_FIGURES[-2],
        "Traces": BOKEH_FIGURES[-1],
    }

    for tool in figures_dict.get("Components").tools:
        if isinstance(tool, BoxZoomTool):
            figures_dict.get("Components").tools.remove(tool)

            figures_dict.get("Components").add_tools(
                BoxZoomTool(match_aspect=True))
        if isinstance(tool, PanTool):
            figures_dict.get("Components").tools.remove(tool)

    # Adjust sizing modes
    figures_dict.get("Components").sizing_mode = "fixed"
    figures_dict.get("Traces").sizing_mode = "stretch_width"

    # Create the layout
    bok_fig = Column(
        figures_dict.get("Slider"),
        Row(
            Column(figures_dict.get("Components"), figures_dict.get("Stats")),
            figures_dict.get("Traces"),
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
        # width=700  # Adjust this value as needed
    )

    # Display in Streamlit
    st.bokeh_chart(bok_fig)


def view_quilt_bokeh():
    figures_dict: dict = {
        "SliderRF": BOKEH_FIGURES[-1],
        "SliderStride": BOKEH_FIGURES[-2],
        "Quilt": BOKEH_FIGURES[-3],
    }

    for tool in figures_dict.get("Quilt").tools:
        if isinstance(tool, BoxZoomTool):
            figures_dict.get("Quilt").tools.remove(tool)

            figures_dict.get("Quilt").add_tools(
                BoxZoomTool(match_aspect=True))
        if isinstance(tool, PanTool):
            figures_dict.get("Quilt").tools.remove(tool)

        # Adjust sizing modes
        figures_dict.get("Quilt").sizing_mode = "fixed"

    # Create the layout
    bokeh_fig = Column(
        figures_dict.get("SliderRF"),
        figures_dict.get("SliderStride"),
        figures_dict.get("Quilt"),
        sizing_mode="scale_width",
        # width=700  # Adjust this value as needed
    )

    # Display in Streamlit
    st.bokeh_chart(bokeh_fig, use_container_width=False)
