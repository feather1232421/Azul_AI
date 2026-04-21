import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def build_iteration_command(run_iteration_path, forwarded_args):
    return [sys.executable, "-u", str(run_iteration_path), *forwarded_args]


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple training iterations by repeatedly invoking run_iteration.py.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to run.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Sleep duration between iterations.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one iteration fails.",
    )
    parser.add_argument(
        "--run-script",
        type=str,
        default="run_iteration.py",
        help="Path to the single-iteration script.",
    )
    args, forwarded_args = parser.parse_known_args()

    run_iteration_path = Path(args.run_script)
    if not run_iteration_path.exists():
        raise FileNotFoundError(f"Run script not found: {run_iteration_path}")

    print(f"Loop start at {datetime.now().isoformat(timespec='seconds')}")
    print(f"Run script: {run_iteration_path}")
    print(f"Iterations: {args.iterations}")
    print(f"Pause seconds: {args.pause_seconds}")
    print(f"Forwarded args: {forwarded_args}")

    failures = []

    for iteration_idx in range(1, args.iterations + 1):
        started_at = time.time()
        command = build_iteration_command(run_iteration_path, forwarded_args)

        print("")
        print(f"[Loop {iteration_idx:03d}/{args.iterations:03d}] Starting at {datetime.now().isoformat(timespec='seconds')}")
        print("Command:", " ".join(command))

        result = subprocess.run(command, check=False)
        elapsed = time.time() - started_at

        print(
            f"[Loop {iteration_idx:03d}/{args.iterations:03d}] "
            f"Finished with exit_code={result.returncode} in {elapsed:.1f}s"
        )

        if result.returncode != 0:
            failures.append(iteration_idx)
            if args.stop_on_error:
                print("Stopping due to --stop-on-error.")
                break

        if iteration_idx < args.iterations and args.pause_seconds > 0:
            print(f"Sleeping for {args.pause_seconds:.1f}s before next iteration.")
            time.sleep(args.pause_seconds)

    print("")
    print(f"Loop end at {datetime.now().isoformat(timespec='seconds')}")
    if failures:
        print(f"Failed iterations: {failures}")
        raise SystemExit(1)

    print("All iterations completed successfully.")


if __name__ == "__main__":
    main()
