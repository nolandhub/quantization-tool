# ABOUTME: Entry point for `python -m qtool`.
# ABOUTME: Delegates to qtool.cli.main and exits with its return code.
import sys

from qtool.cli import main

sys.exit(main())
