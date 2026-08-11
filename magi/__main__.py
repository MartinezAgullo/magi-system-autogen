"""``python -m magi`` — delegates to :mod:`magi.main`.

Kept separate so the boot logic lives in an importable module (tests, and the
``magi`` console script) rather than in a file that only runs as a script.
"""

from magi.main import main

if __name__ == "__main__":
    main()
