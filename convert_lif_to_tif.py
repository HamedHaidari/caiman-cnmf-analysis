import multiprocessing as mp
from pathlib import Path

from bioio import BioImage


def convert_lif_to_tif(lif_file, output_dir):
    image = BioImage(lif_file)

    tif_dir = output_dir / lif_file.stem
    tif_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(lif_file).stem

    for scene in image.scenes:
        tif_path = tif_dir / f"{base_name}__{scene}.tif"
        image.save(tif_path, select_scenes=[scene])

    print(f"Converted {lif_file} to .tif")


def main():
    input_dir = Path("/workspaces/Caiman_Analysis/data")
    output_dir = Path("/workspaces/Caiman_Analysis/data")

    lif_files = list(input_dir.glob("*.lif"))

    # Create a pool of worker processes
    with mp.Pool(processes=mp.cpu_count()-20) as pool:
        # Map the conversion function to all .lif files
        pool.starmap(convert_lif_to_tif, [
                     (file, output_dir) for file in lif_files])


if __name__ == "__main__":
    main()
