"""Zero-Mem: Zero-Token Memory Operations for LLM Agents.

Usage:
    uv run python -m zero_mem.index_builder  # build index from messages
    uv run python -m zero_mem.retriever       # retrieve evidence for a query

Or programmatic:
    from zero_mem.index_builder import build_index
    from zero_mem.retriever import build_retriever
    from zero_mem.calibration import CalibrationLayer
"""

import sys


def main():
    print("Zero-Mem v0.1.0 — Zero-Token Memory Operations for LLM Agents")
    print("")
    print("Modules available:")
    print("  zero_mem.index_builder  — Build dual-view index from messages")
    print("  zero_mem.retriever      — Dual-view retrieval with evidence closure")
    print("  zero_mem.calibration    — Deterministic calibration (filter/rank/calibrate)")
    print("")
    print("Skill: zero-mem-retrieval — read_skill('zero-mem-retrieval') for full docs")


if __name__ == "__main__":
    main()
