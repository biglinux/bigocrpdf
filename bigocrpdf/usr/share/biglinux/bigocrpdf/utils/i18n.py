"""
Translation utility module to ensure consistent translations throughout the application
"""

import gettext

gettext.textdomain("bigocrpdf")
_ = gettext.gettext
