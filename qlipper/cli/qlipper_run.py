import argparse

from qlipper.configuration import SimConfig
from qlipper.constants import CFG_DIR
from qlipper.postprocess import postprocess_run
from qlipper.run.mission_runner import run_mission


def main():
    parser = argparse.ArgumentParser(description="Run a qlipper mission.")
    parser.add_argument("--cfg", type=str, help="Path to config file.", required=False)

    args = parser.parse_args()

    if args.cfg is None:
        from tkinter.filedialog import askopenfilename

        args.cfg = askopenfilename(
            initialdir=CFG_DIR,
            title="Select config file",
            filetypes=(("Qlipper JSON Config", "*.json"), ("all files ", "*.*")),
        )

    cfg = SimConfig.from_file(args.cfg)

    run_id, t, y = run_mission(cfg)
    postprocess_run(run_id, t, y, cfg)


if __name__ == "__main__":
    main()
