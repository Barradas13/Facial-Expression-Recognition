"""Register this Poetry virtualenv as a Jupyter kernel.

Why this exists
---------------
The notebooks in ``OSEMN/`` only need ``ipykernel`` (plus the project
dependencies) to be executed cell by cell. ``ipykernel`` is already a dev
dependency of ``pyproject.toml``, but the virtualenv Poetry creates lives in its
own cache folder, and VS Code does not always surface it in the kernel picker
(the Poetry locator uses ``poetry env info --path``, which misreports when a
newer Python sits on ``PATH``). Registering an explicit kernelspec makes the
environment selectable no matter what the automatic discovery does.

The kernelspec points at *this* interpreter, i.e. the virtualenv Poetry
resolved from ``pyproject.toml`` / ``poetry.lock`` - nothing is installed and
the environment is not modified.

Usage (from ``OSEMN/``):

    poetry run python tools/setup_kernel.py            # install / refresh
    poetry run python tools/setup_kernel.py --check    # show existing spec
    poetry run python tools/setup_kernel.py --remove   # unregister again
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

# Name used by the notebook UIs ("Select Kernel" -> "Jupyter Kernel...").
KERNEL_NAME = "osemn-obtain-fer"
DISPLAY_NAME = "Python (OSEMN / Poetry)"

# Imported by the code cells of the two notebooks; reported for information only.
NOTEBOOK_MODULES = (
    "cv2",
    "mediapipe",
    "numpy",
    "pandas",
    "tqdm",
    "matplotlib",  # used by explore_fer.ipynb (resolved through mediapipe)
)


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _in_virtualenv() -> bool:
    """True when the interpreter is a virtualenv, not the system/base Python."""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _missing_modules() -> List[str]:
    return [name for name in NOTEBOOK_MODULES if importlib.util.find_spec(name) is None]


def _spec_json(kernel_name: str) -> Optional[Path]:
    """Return the kernelspec.json path for ``kernel_name`` if it is installed."""
    try:
        from jupyter_client.kernelspec import KernelSpecManager
    except ImportError:  # pragma: no cover - ipykernel always brings jupyter_client
        return None
    try:
        spec = KernelSpecManager().get_kernel_spec(kernel_name)
    except Exception:
        return None
    return Path(spec.resource_dir) / "kernel.json"


def do_install() -> int:
    if not _in_virtualenv():
        return _fail(
            "this is the base interpreter, not the Poetry virtualenv.\n"
            "       Run it through Poetry:  poetry run python tools/setup_kernel.py"
        )

    try:
        from ipykernel import kernelspec
    except ImportError:
        return _fail(
            "ipykernel is not installed in this environment.\n"
            "       Fix with:  poetry install --with dev   (declared in [tool.poetry.group.dev])"
        )

    if sys.version_info[:2] != (3, 12):
        print(
            "warning: pyproject.toml targets 3.11-3.12, this interpreter is "
            f"{sys.version.split()[0]}",
            file=sys.stderr,
        )

    missing = _missing_modules()
    if missing:
        print(
            f"warning: not importable in this env: {', '.join(missing)}\n"
            "         run `poetry install` before executing the notebooks",
            file=sys.stderr,
        )

    # Installs <user-data-dir>/kernels/osemn-obtain-fer/kernel.json pointing at
    # this interpreter (`<venv>/Scripts/python.exe -m ipykernel_launcher -f ...`).
    kernelspec.install(kernel_name=KERNEL_NAME, user=True, display_name=DISPLAY_NAME)

    spec_path = _spec_json(KERNEL_NAME)
    print()
    print(f"kernel name : {KERNEL_NAME}")
    print(f"display name: {DISPLAY_NAME}")
    print(f"interpreter : {sys.executable}")
    print(f"kernelspec  : {spec_path if spec_path else 'unknown'}")
    print()
    print("Use it in VS Code: open the .ipynb, click the kernel picker (top right)")
    print(f"-> Select Another Kernel -> Jupyter Kernel... -> {DISPLAY_NAME}.")
    print("The kernel starts in the notebook's folder, so ./data/images and ./outputs resolve.")
    return 0


def do_check() -> int:
    spec_path = _spec_json(KERNEL_NAME)
    if spec_path is None or not spec_path.exists():
        print(f"kernel '{KERNEL_NAME}' is NOT registered")
        print("register it with: poetry run python tools/setup_kernel.py")
        return 1

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    print(f"kernel '{KERNEL_NAME}' is registered")
    print(f"  kernelspec : {spec_path}")
    print(f"  display    : {spec.get('display_name')}")
    print(f"  language   : {spec.get('language')}")
    print(f"  argv       : {' '.join(spec.get('argv', []))}")
    exists = Path(spec.get("argv", ["", ""])[0]).exists()
    print(f"  executable : {'OK' if exists else 'MISSING (recreate the env / re-run --install)'}")
    return 0 if exists else 1


def do_remove() -> int:
    try:
        from ipykernel import kernelspec
    except ImportError:
        return _fail("ipykernel is not installed in this environment")
    kernelspec.remove(kernel_name=KERNEL_NAME, user=True)
    print(f"removed kernelspec '{KERNEL_NAME}'")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register the OSEMN Poetry virtualenv as a Jupyter kernel "
        "(so notebook cells run in exactly the locked environment)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="show the registered kernelspec")
    group.add_argument("--remove", action="store_true", help="unregister the kernelspec")
    args = parser.parse_args(argv)

    if args.check:
        return do_check()
    if args.remove:
        return do_remove()
    return do_install()


if __name__ == "__main__":
    raise SystemExit(main())
