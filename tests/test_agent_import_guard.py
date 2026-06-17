"""
tests/test_agent_import_guard.py -- the CPU agent layer must stay GPU-dep-free.

agentic_cpu consumes structured detection JSON; it must NOT import torch,
torchvision, ultralytics, cv2, transformers, or any vision loader. We assert this
in a CLEAN subprocess (so unrelated imports in the test session cannot mask a
real leak).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

FORBIDDEN = ["torch", "torchvision", "ultralytics", "cv2", "transformers",
             "yolo26_loader", "edgecrafter_loader", "official_deimv2_loader"]

_PROBE = """
import sys
import agentic_cpu
from agentic_cpu import router, agents, approvals, jobs, action_log, graph, llm, config, schemas
from agentic_cpu.agents import (company_setup, safety_observation, risk_assessment,
    audit_writer, capa_writer, training_writer, vision_improvement)
from agentic_cpu.tools import (vision_tools, document_tools, risk_tools,
    audit_tools, capa_tools, training_tools)
forbidden = {forbidden!r}
leaked = sorted(m for m in forbidden if m in sys.modules)
print("LEAKED:" + ",".join(leaked))
sys.exit(1 if leaked else 0)
"""


def test_agentic_cpu_imports_no_gpu_modules():
    code = _PROBE.format(forbidden=FORBIDDEN)
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                          capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    assert proc.returncode == 0, f"CPU agent leaked GPU deps -> {out}"
    assert "LEAKED:" in proc.stdout
    assert proc.stdout.strip().endswith("LEAKED:")   # nothing after the colon


def test_agentic_cpu_router_mounts_without_gpu():
    """Importing + building the router must not require any GPU dep either."""
    code = (
        "import agentic_cpu, sys\n"
        "r = agentic_cpu.get_router()\n"
        "paths = sorted({route.path for route in r.routes})\n"
        "assert any(p.endswith('/health') for p in paths), paths\n"
        "bad = [m for m in " + repr(FORBIDDEN) + " if m in sys.modules]\n"
        "print('BAD:' + ','.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                          capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)
