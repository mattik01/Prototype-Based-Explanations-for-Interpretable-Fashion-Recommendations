# S0-build test harness — shared check() helper + repo-root path setup.
# Imported by the t0x/i0x checks in this directory. Not a test itself.
# (Mirrors dc01/dc02 _harness.py conventions.)
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def make_check():
    fails = []

    def check(name, cond, evidence=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {evidence}" if evidence else ""))
        if not cond:
            fails.append(name)

    return check, fails
