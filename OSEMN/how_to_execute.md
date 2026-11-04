# How to execute the notebooks

Everything here runs in the Poetry environment declared by `OSEMN/pyproject.toml`
and locked in `OSEMN/poetry.lock` (Python 3.11–3.12; resolved to **3.12.10** on
this machine). Nothing is installed globally.

```powershell
cd OSEMN
poetry install --with dev                 # once per machine: builds the venv from the lockfile
poetry run python tools/setup_kernel.py   # once per machine: registers it as a Jupyter kernel
```

Then open a notebook in VS Code, pick the kernel **Python (OSEMN / Poetry)**
and run any cell you want. No other step is required.

---

## 1. What the environment is

| | |
|---|---|
| Project | `OSEMN/pyproject.toml` (`requires-python = ">=3.11,<3.13"`) |
| Venv | `%LOCALAPPDATA%\pypoetry\Cache\virtualenvs\osemn-obtain-fer-MP28oxs9-py3.12` |
| Interpreter | `...\osemn-obtain-fer-MP28oxs9-py3.12\Scripts\python.exe` (3.12.10) |
| Notebook deps | `mediapipe`, `numpy`, `opencv-contrib-python` (cv2), `pandas`, `tqdm`, `matplotlib` |
| Cell execution | `ipykernel` — declared in `[tool.poetry.group.dev.dependencies]` |

Ask Poetry where the venv is:

```powershell
poetry env list --full-path
```

> `poetry env info --path` is unreliable here: Poetry 2.4.1 sees the newer Python
> on `PATH` (3.14.7) and reports `Path: NA`. Use `poetry env list --full-path`
> or the line printed by `setup_kernel.py` instead.

The message `The currently activated Python version 3.14.7 is not supported by
the project ... Using python.exe (3.12.10)` on every `poetry run` is expected:
Poetry is picking the interpreter the project allows. It is not an error.

---

## 2. VS Code — run single cells, ranges or everything

1. Open `OSEMN/obtain_fer.ipynb` or `OSEMN/explore_fer.ipynb`.
2. Kernel picker (top right) → **Select Another Kernel…** →
   **Jupyter Kernel…** → **Python (OSEMN / Poetry)**.
   The Poetry cache venv is not guaranteed to appear under *Python
   Environments* (VS Code discovers Poetry envs with `poetry env info --path`,
   which misreports — see above), so the registered kernelspec is the reliable
   entry. It is listed by `poetry run jupyter kernelspec list` as
   `osemn-obtain-fer`.
3. Run what you need:

| Action | How |
|---|---|
| Run cell, move to next | `Shift+Enter` |
| Run cell, stay | `Ctrl+Enter` |
| Run cell, insert one below | `Alt+Enter` |
| Run only this cell (menu) | cell `▾` toolbar → *Run Cell* |
| Run everything above / below | cell `▾` toolbar → *Run All Above* / *Run All Below* |
| Run everything in order | toolbar → *Run All* (or command palette → *Notebook: Run All*) |
| Clear state | toolbar → *Restart*; then run the cells you need again |

### Things worth knowing

* **State is per kernel session.** Cells are independent text, but they share
  one Python process. A cell that uses `CONFIG`, `validate_image`, …, needs the
  cell that defines them to have run first in the same session. `Restart` wipes
  that; *Run All Above* rebuilds it.
* **Working directory = the notebook's folder** (`OSEMN/`). VS Code's
  `jupyter.notebookFileRoot` defaults to `${fileDirname}` and is pinned to it in
  `.vscode/settings.json`. That is what makes the relative paths in the
  notebooks resolve:
  * `obtain_fer.ipynb` reads `./data/images`, writes `./outputs`
  * `explore_fer.ipynb` reads `./outputs` and writes only
    `./outputs/explore_audit.csv` — its figures are rendered inline, at the end
    of each cell, and never saved to disk

  If `Path.cwd()` is not `.../OSEMN`, those relative paths point elsewhere.
* **Only the cells you run have effects.** In `obtain_fer.ipynb` every cell
  except the demo (cell index 29) just defines functions/constants, and the demo
  call itself is commented out — running the whole notebook writes nothing. To
  actually produce the tables, uncomment the demo cell or run a cell containing:
  ```python
  results = run_obtain_pipeline(image_dir="./data/images", output_dir="./outputs",
                                save_crops=False, include_z=False)
  ```
* `explore_fer.ipynb` needs the `./outputs/*.csv` files from a previous Obtain
  run; when they are missing it raises `FileNotFoundError` naming the missing ones.

---

## 3. Terminal — run exactly the cells you pick (no editor)

`tools/run_cells.py` is the headless equivalent of clicking cells: it runs the
cells you list, in the order you list them, in one shared namespace, with the
cwd set to the notebook's folder and matplotlib on the non-interactive `Agg`
backend (so `plt.show()` cannot block).

```powershell
cd OSEMN

# see every code cell and its index (the number you pass to --cells)
poetry run python tools/run_cells.py --list
poetry run python tools/run_cells.py obtain_fer.ipynb --list

# one cell / several cells / a range / the whole notebook
poetry run python tools/run_cells.py obtain_fer.ipynb --cells 3
poetry run python tools/run_cells.py obtain_fer.ipynb --cells 3,5,7
poetry run python tools/run_cells.py obtain_fer.ipynb --cells 3-9
poetry run python tools/run_cells.py explore_fer.ipynb --cells 2,4
poetry run python tools/run_cells.py explore_fer.ipynb --cells all

# pick cells by their content instead of by index, and add the cells they need
poetry run python tools/run_cells.py explore_fer.ipynb --grep "n_rejected" --cells 2

# keep going after a cell raises / let matplotlib open windows
poetry run python tools/run_cells.py explore_fer.ipynb --cells all --keep-going --gui
```

* Indices are the raw `.ipynb` indices (markdown cells included) — exactly what
  `--list` prints, next to the "code cell number". A range like `5-8` runs the
  code cells inside it and skips markdown positions; a single index must be a
  code cell (typos are reported with the valid list).
* Order matters and is respected: `--cells 9,3` really runs 9 then 3.
* A trailing expression in a cell is printed, like IPython does.
* Exit code `0` on success, `1` when a cell raised, `2` on a bad selection.
* A cell that depends on earlier ones fails cleanly when run alone, and the
  runner prints the traceback of the cell (not of itself) plus the command that
  includes its prerequisites.

From outside `OSEMN/` either `cd OSEMN` first or point Poetry at the project:

```powershell
poetry -P OSEMN run python OSEMN/tools/run_cells.py obtain_fer.ipynb --cells 3
```

---

## 4. The kernel, and how to repair it

```powershell
poetry run python tools/setup_kernel.py           # install / refresh (idempotent)
poetry run python tools/setup_kernel.py --check   # show the kernelspec + interpreter
poetry run python tools/setup_kernel.py --remove  # unregister
poetry run jupyter kernelspec list                # every kernel Jupyter knows about
```

The kernelspec is `%APPDATA%\jupyter\kernels\osemn-obtain-fer\kernel.json` and
its `argv[0]` is the venv's `python.exe`, so a cell always runs in the locked
environment (it does not depend on VS Code's interpreter picker).

**Kernel not in the list?**
1. `poetry run python tools/setup_kernel.py --check` — if it says *NOT
   registered*, run it without `--check`.
2. Command palette → *Developer: Reload Window* (the kernel list is cached).
3. Kernel picker → *Select Another Kernel…* → *Jupyter Kernel…*.

**Venv was deleted or moved?** `poetry install --with dev` recreates it (new
hash suffix in the folder name), then re-run `setup_kernel.py`. Its printed
`interpreter :` line is the path to put in `.vscode/settings.json`
(`python.defaultInterpreterPath`, which is machine-specific; the kernel
registration is the portable part).

**VS Code keeps choosing another Python?** `python.defaultInterpreterPath` in
`.vscode/settings.json` only applies until you pick an interpreter manually;
the notebook kernel and the interpreter are separate choices — the kernel you
pick in the notebook wins.

---

## 5. Optional: JupyterLab / classic notebook UI

Not installed by default (it is not needed for VS Code). If you want it, install
it into the same project and use the already-registered kernel:

```powershell
poetry add --group dev jupyterlab
poetry run jupyter lab     # kernel "Python (OSEMN / Poetry)" is in the launcher
```

Because `jupyter lab` is started from `OSEMN/`, kernels start there too, so the
notebooks' relative paths keep working.
