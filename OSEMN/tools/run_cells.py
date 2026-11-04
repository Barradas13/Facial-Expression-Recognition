"""Execute exactly the notebook cells you ask for, inside the Poetry env.

Interactive cell-by-cell execution in VS Code is the normal workflow (see
``how_to_execute.md``). This is the headless equivalent, useful for CI, for
re-running a subset of cells after an edit, or when no editor is available:

* cells run **in the order you list them**, in one shared namespace, so state
  (functions, dataframes, ``CONFIG``...) carries over exactly like in a kernel;
* the process ``cwd`` is the notebook's own folder, which is what makes the
  relative paths in the notebooks (``./data/images``, ``./outputs``) resolve;
* the value of a trailing expression is printed, like IPython does;
* matplotlib uses the non-interactive ``Agg`` backend so ``plt.show()`` cannot
  block the run (``--gui`` keeps whatever backend the env would normally use).

Usage (run from ``OSEMN/`` so Poetry picks the right environment):

    poetry run python tools/run_cells.py --list
    poetry run python tools/run_cells.py obtain_fer.ipynb --list
    poetry run python tools/run_cells.py obtain_fer.ipynb --cells 3,5-8
    poetry run python tools/run_cells.py explore_fer.ipynb --cells 2,4
    poetry run python tools/run_cells.py explore_fer.ipynb --grep "MANIFEST"

Cell indices are the raw indices in the ``.ipynb`` JSON (markdown cells
included), i.e. exactly the numbers printed by ``--list``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def notebook_files(directory: Path) -> List[Path]:
    """All .ipynb files in ``directory``, in a stable order."""
    return sorted(path for path in directory.glob("*.ipynb") if path.is_file())


def resolve_notebook(name: str, directory: Path) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = (directory / path).resolve()
    if path.suffix != ".ipynb" or not path.is_file():
        raise FileNotFoundError(f"notebook not found: {path}")
    return path


def cell_source(cell: Dict) -> str:
    """Notebook cells store source either as a list of lines or as one string."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def code_cells(notebook: Dict) -> List[Tuple[int, int, str]]:
    """``(raw_index, code_ordinal, source)`` for every code cell."""
    cells = []
    ordinal = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        ordinal += 1
        cells.append((index, ordinal, cell_source(cell)))
    return cells


def preview(source: str, width: int = 78) -> str:
    """First meaningful line of a cell, for the --list output."""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return (stripped[: width - 1] + "...") if len(stripped) > width else stripped
    return ""


def parse_selection(spec: str, available: Sequence[int]) -> List[int]:
    """Turn ``"3,5-8"`` (or ``"all"``) into a validated list of raw cell indices.

    Indices are raw ``.ipynb`` indices (markdown cells included), which is what
    ``--list`` shows. A range covers the code cells inside it and silently skips
    markdown positions, so ``5-8`` works on a notebook whose code cells are only
    at odd indices. A single index must be a code cell, so typos are caught.
    Ranges keep the order they were written in, so ``--cells 9,3`` really runs 9
    then 3 - handy when a later cell defines what an earlier one needs.
    """
    available_set = set(available)
    known = ", ".join(str(index) for index in available)
    selection: List[int] = []
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    if not tokens:
        raise ValueError("empty cell selection")

    for token in tokens:
        if token.lower() in ("all", "*"):
            selection.extend(sorted(available_set))
            continue
        if "-" in token or ".." in token:
            separator = "-" if "-" in token else ".."
            start_text, _, end_text = token.partition(separator)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError:
                raise ValueError(f"invalid range {token!r}; expected e.g. 5-8") from None
            if start > end:
                raise ValueError(f"invalid range {token!r}; start must be <= end")
            inside = [index for index in range(start, end + 1) if index in available_set]
            if not inside:
                raise ValueError(
                    f"range {token!r} contains no code cells. Code cells are: {known}"
                )
            selection.extend(inside)
            continue
        try:
            index = int(token)
        except ValueError:
            raise ValueError(f"invalid cell index {token!r}") from None
        if index not in available_set:
            raise ValueError(
                f"cell {index} is not a code cell of this notebook. Code cells are: {known}"
            )
        selection.append(index)

    return selection


def select_by_pattern(pattern: str, cells: Sequence[Tuple[int, int, str]]) -> List[int]:
    """Raw indices of code cells whose source contains ``pattern`` (case-insensitive)."""
    needle = pattern.lower()
    return [index for index, _ordinal, source in cells if needle in source.lower()]


def run_source(source: str, namespace: Dict[str, object], filename: str) -> None:
    """Execute one cell the way IPython does (trailing expression gets displayed)."""
    tree = ast.parse(source, filename=filename)
    trailing = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        trailing = ast.Expression(tree.body[-1].value)
        tree.body.pop()

    exec(compile(tree, filename, "exec"), namespace)  # noqa: S102 - deliberate cell execution

    if trailing is not None:
        value = eval(compile(trailing, filename, "eval"), namespace)  # noqa: S307
        if value is not None:
            print(repr(value))


def print_cell_traceback(error: BaseException) -> None:
    """Print the failing cell's traceback without this runner's own frames."""
    tb = error.__traceback__
    runner = Path(__file__).resolve()
    while tb is not None and Path(tb.tb_frame.f_code.co_filename).resolve() == runner:
        tb = tb.tb_next
    if tb is None:
        print(f"{type(error).__name__}: {error}")
    else:
        traceback.print_exception(type(error), error, tb)


def print_listing(paths: Sequence[Path]) -> None:
    """``--list``: show every code cell with the index to pass to ``--cells``."""
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = code_cells(notebook)
        print(f"\n{path.name}  ({len(cells)} code cells)")
        print(f"  {'cell':>4}  {'code':>4}  first line")
        print(f"  {'-' * 4}  {'-' * 4}  {'-' * 60}")
        for index, ordinal, source in cells:
            print(f"  {index:>4}  {ordinal:>4}  {preview(source)}")
    print(
        "\nRun them with:  poetry run python tools/run_cells.py "
        "<notebook> --cells 3,5-8     (or --cells all)"
    )


def execute_cells(
    notebook: Path,
    cells: Sequence[Tuple[int, int, str]],
    selection: Sequence[int],
    keep_going: bool,
) -> int:
    """Run the selected cells in order; returns a process exit code."""
    sources = {index: (ordinal, source) for index, ordinal, source in cells}
    namespace: Dict[str, object] = {"__name__": "__main__"}
    failures = 0

    print(f"notebook : {notebook}")
    print(f"cwd      : {Path.cwd()}")
    print(f"python   : {sys.executable}")
    print(f"cells    : {', '.join(str(index) for index in selection)}")
    print()

    for position, index in enumerate(selection, start=1):
        ordinal, source = sources[index]
        print(f"[{position}/{len(selection)}] cell {index} (code #{ordinal}) ... ", end="", flush=True)
        started = time.perf_counter()
        try:
            run_source(source, namespace, f"{notebook.name}:cell {index}")
        except KeyboardInterrupt:
            print("interrupted")
            return 130
        except BaseException as error:  # noqa: BLE001 - report anything the cell raises
            failures += 1
            print(f"FAILED after {time.perf_counter() - started:.2f}s")
            print_cell_traceback(error)
            print()
            print(f"cell {index} raised. A cell usually needs the ones above it, so include them:")
            print(
                f"  poetry run python tools/run_cells.py {notebook.name} "
                f"--cells {cells[0][0]}-{index}"
            )
            print(f"  poetry run python tools/run_cells.py {notebook.name} --cells all")
            if not keep_going:
                return 1
            continue
        print(f"ok ({time.perf_counter() - started:.2f}s)")

    print()
    if failures:
        print(f"{failures} of {len(selection)} cell(s) failed")
        return 1
    print(f"all {len(selection)} cell(s) ran without errors")
    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selected cells of a Jupyter notebook in the Poetry environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples (from OSEMN/):\n"
            "  poetry run python tools/run_cells.py --list\n"
            "  poetry run python tools/run_cells.py obtain_fer.ipynb --cells 3\n"
            "  poetry run python tools/run_cells.py obtain_fer.ipynb --cells 3,5-8\n"
            "  poetry run python tools/run_cells.py explore_fer.ipynb --cells all\n"
            '  poetry run python tools/run_cells.py explore_fer.ipynb --grep "n_rejected" --cells 2\n'
        ),
    )
    parser.add_argument(
        "notebook",
        nargs="?",
        help="notebook to run (relative to the current folder or absolute); "
        "omit it with --list to show every notebook in OSEMN/",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_cells",
        help="list code cells and their indices instead of executing",
    )
    parser.add_argument(
        "--cells",
        metavar="SPEC",
        help="cells to run: comma separated indices/ranges as printed by --list, "
        "e.g. 3,5-8 or all (markdown positions inside a range are skipped)",
    )
    parser.add_argument(
        "--grep",
        metavar="PATTERN",
        help="run every code cell whose source contains PATTERN (case-insensitive); "
        "combine with --cells to add the cells that define what they need",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="keep the default matplotlib backend (figures may open windows and block)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="carry on with the next cell when one raises",
    )
    return parser


def find_notebook(name: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    """Resolve a notebook given on the command line (cwd first, then OSEMN/)."""
    for directory in search_dirs:
        try:
            return resolve_notebook(name, directory)
        except FileNotFoundError:
            continue
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    here = Path(__file__).resolve().parent.parent  # OSEMN/

    if args.list_cells and not args.notebook:
        found = notebook_files(here)
        if not found:
            print(f"no .ipynb found in {here}", file=sys.stderr)
            return 1
        print_listing(found)
        return 0

    if not args.notebook:
        print(
            "error: give a notebook, or use --list to see the available ones",
            file=sys.stderr,
        )
        return 2

    notebook = find_notebook(args.notebook, (Path.cwd(), here))
    if notebook is None:
        print(f"error: notebook not found: {args.notebook}", file=sys.stderr)
        return 1

    cells = code_cells(json.loads(notebook.read_text(encoding="utf-8")))

    if args.list_cells:
        print_listing([notebook])
        return 0

    if not cells:
        print(f"error: {notebook} has no code cells", file=sys.stderr)
        return 1

    available = [index for index, _ordinal, _source in cells]
    try:
        if args.grep:
            selection = select_by_pattern(args.grep, cells)
            if not selection:
                print(f"error: no code cell contains {args.grep!r}", file=sys.stderr)
                return 1
            if args.cells:
                # Union in notebook order: "these cells, plus anything that mentions X".
                selection = sorted(set(selection) | set(parse_selection(args.cells, available)))
        else:
            selection = parse_selection(args.cells or "all", available)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    # Relative paths inside the notebooks resolve against the notebook folder.
    os.chdir(notebook.parent)
    if str(notebook.parent) not in sys.path:
        sys.path.insert(0, str(notebook.parent))
    if not args.gui:
        os.environ.setdefault("MPLBACKEND", "Agg")
        # Cells end with plt.show(); with Agg that is a headless no-op and matplotlib
        # warns about it once per call - drop that noise, keep every other warning.
        warnings.filterwarnings("ignore", message=".*non-interactive, and thus cannot be shown.*")

    return execute_cells(notebook, cells, selection, args.keep_going)


if __name__ == "__main__":
    raise SystemExit(main())
