"""Enables ``python -m aiperfopt`` as an equivalent to the ``aiperfopt`` script."""

from aiperfopt.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
