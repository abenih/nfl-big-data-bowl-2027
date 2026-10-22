"""
Wrapper to execute build_notebook_light.py.
"""
import subprocess
import sys

if __name__ == '__main__':
    subprocess.run([sys.executable, '/home/benji/Downloads/nfl_bdb_2027_bend/src/build_notebook_light.py'], check=True)
