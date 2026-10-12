"""Report dataset availability for the code-only release; no network access."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Show dataset availability")
    parser.add_argument("--groups", help="Reserved for a future dataset release")
    parser.parse_args(argv)
    print("Dataset: In preparation. This code release does not distribute or download benchmark data.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
