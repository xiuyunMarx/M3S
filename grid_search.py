import subprocess
import os
import sys


def run_grid_search():
    # cfg_values = [5.0, 10.0, 15.0, 20.0, 25.0]
    # swap_guidance_scales = [5.0, 10.0, 15.0, 20.0, 25.0]
    # configs = [(10, 15), (20, 15), (15, 10), (15, 20)]
    # configs = [(20, 20)]
    # configs = [(17.5, 15), (17.5, 17.5), (17.5, 20), (20, 10)]
    configs = [(7.5, 7.5)]

    # The directory where Generation_demo.py is located
    working_dir = "SDv1.5"  

    # Check if the directory exists
    if not os.path.exists(working_dir):
        print(f"Error: Directory '{working_dir}' not found.")
        print("Please ensure you are running this script from the project root (M3S).")
        sys.exit(1)

    # print(
    #     f"Starting Grid Search over {len(cfg_values) * len(swap_guidance_scales)} combinations..."
    # )

    # for cfg in cfg_values:
    #     for swap_scale in swap_guidance_scales:
    for cfg, swap_scale in configs:
        print(f"\n[Grid Search] Running: CFG={cfg}, Swap Guidance Scale={swap_scale}")

        # Construct the command
        # python -u Generation_demo.py --cfg <val> --swap_guidance_scale <val>
        cmd = [
            "python",
            "-u",
            "Generation_demo.py",
            "--cfg",
            str(cfg),
            "--swap_guidance_scale",
            str(swap_scale),
        ]

        try:
            # Run the command inside the SDv1.5 directory
            subprocess.run(cmd, cwd=working_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"Error occurred while running combination CFG={cfg}, Scale={swap_scale}"
            )
            print(e)
        except KeyboardInterrupt:
            print("\nGrid search interrupted by user.")
            sys.exit(0)

    print("\nGrid search completed.")


if __name__ == "__main__":
    run_grid_search()
