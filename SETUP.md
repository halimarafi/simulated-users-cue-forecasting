# Running the project locally

Tested with Python 3.11. Commands are for a Mac terminal (VS Code: Terminal → New Terminal).

## 1. One-time setup

```bash
cd path/to/seldon_project            # the unzipped folder
xcode-select --install               # Mac only: C++ compiler for Stan. Skip if already installed.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m cmdstanpy.install_cmdstan  # builds Stan, about 5-15 minutes
```

In VS Code, open any notebook, click **Select Kernel** (top right), and choose the `.venv` Python.

## 2. Run order

| Notebook | What it does | Expected time |
| --- | --- | --- |
| `01_data.ipynb` | Loads the data, applies exclusions, saves `data/derived/trials_clean.csv`, draws descriptive plots | Under a minute |
| `02a_rung2_test_fit.ipynb` | Short timing run of rung 2. Prints how long a full fit will take on your machine | Minutes |
| `02_models_fit.ipynb` | Fits rungs 0, 1 and 2 to basic trials | Rungs 0 and 1: a few minutes each. Rung 2: see 02a |
| `03_model_checks.ipynb` | Convergence, basic-trial match, parameter recovery | About as long as 02 (it refits rung 2) |

Notebook 01 reads the data from the path set in its first code cell. Change `DATA_PATH` there if the file moves.

Notebooks 02 and 03 have a `QUICK` switch at the top. `QUICK = True` runs a 5-minute version on 8 people to check everything works. Its numbers mean nothing. Set it back to `False` for the real run.

## 3. What to send back

- The diagnostics table printed at the end of the fitting cell in 02.
- The recovery table in 03.
- The timing line from 02a.
