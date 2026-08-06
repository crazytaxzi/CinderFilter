"""Compatibility module for the packaged build.

The packaged cinderfilter.py already contains the PyO3 thread-affinity fix. This
module keeps the same extension import surface used by the GitHub build.
"""

import cinderfilter as cf

AudioEngine = cf.AudioEngine
CinderFilterApp = cf.CinderFilterApp
main = cf.main

if __name__ == "__main__":
    cf.main()
