#!/usr/bin/env python3
"""
BigOcrPdf - Entry point for python -m bigocrpdf

This module allows the package to be run as a module:
    python -m bigocrpdf
"""

import sys

from bigocrpdf import main

if __name__ == "__main__":
    sys.exit(main())
