"""Isolated exact text/PIC binding API, intended for loopback port 8014."""

from experiments.evidence_coverage_exact import run_case_exact

import api_server_vnext_fast as base


base.run_case_fast = run_case_exact
base.app.title = "Manual Retrieval Vnext Exact (isolated)"
base.app.version = "0.2.0"
app = base.app
