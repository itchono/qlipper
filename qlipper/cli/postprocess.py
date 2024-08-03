import argparse
from pathlib import Path

from qlipper.constants import OUTPUT_DIR
from qlipper.postprocess import postprocess_from_folder


def main():
    parser = argparse.ArgumentParser(description="Postprocess a qlipper run.")
    parser.add_argument(
        "--folder", type=str, help="Folder containing the run.", required=False
    )

    args = parser.parse_args()

    if args.folder is None:
        from tkinter.filedialog import askdirectory

        args.folder = askdirectory(initialdir=OUTPUT_DIR, title="Select case folder")

    postprocess_from_folder(Path(args.folder))


if __name__ == "__main__":
    main()
