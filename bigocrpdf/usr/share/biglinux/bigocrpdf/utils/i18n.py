#!/usr/bin/env python3
"""
BigOcrPdf - Internationalization Module

This module initializes gettext for internationalization support
"""

import gettext

gettext.textdomain("bigocrpdf")
_ = gettext.gettext
