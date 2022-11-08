"""
a script for wrapping around my python scripts
"""
import os
import time
import sys
import argparse
import logging
import subprocess


def main(args=sys.argv[1:]):
    is_debug = args[0] == "local"
    target_file = args[1]
    run_line = " ".join(args[2:])
    if is_debug:
        subprocess.check_output(
            "python %s" % run_line, stderr=subprocess.STDOUT, shell=True
        )
    else:
        output = subprocess.check_output(
            "qsub -cwd run_script.sh %s" % run_line,
            stderr=subprocess.STDOUT,
            shell=True,
        )
        print("QSUB DONE", output)

    if not is_debug:
        # This code is really simple. Wait at most 100 * 10 seconds until the
        # desired result pops up in the file system.
        for i in range(200):
            if not os.path.exists(target_file):
                time.sleep(20)
            else:
                break

        time.sleep(20)


if __name__ == "__main__":
    main(sys.argv[1:])
