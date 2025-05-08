#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BigOcrPdf - Internationalization Module

This module initializes gettext for internationalization support
"""

import gettext

# Configure the translation domain
gettext.textdomain("bigocrpdf")

# Export _ directly as the translation function
_ = gettext.gettext