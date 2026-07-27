# Machining Tool Life Analysis

A Python command-line tool for exploratory comparison of segmented machine-tool sensor signals recorded under baseline and simulated fault conditions.

The software was developed for the Data Science and Research Software Project module at the University of Sheffield.

## Dataset

The project uses the following publicly available dataset:

> Dominguez Caballero, J. A., Moore, J., and Stammers, J. (2023). *Sensor signals for machine tool and process health assessment*. The University of Sheffield. Dataset.

DOI:

```text
https://doi.org/10.15131/shef.data.24125715.v1
```

Dataset page:

```text
https://orda.shef.ac.uk/articles/dataset/Sensor_signals_for_machine_tool_and_process_health_assessment_/24125715
```

The dataset is not included in this repository. The data files are large and must be downloaded separately from the official source.

Place the downloaded `.mat` files inside a local directory called:

```text
data/
```

The `data/` directory and common HDF5 data-file extensions are excluded through `.gitignore`.

## Purpose

The software loads segmented sensor signals from MATLAB v7.3/HDF5 files and compares measurements from two experimental conditions, such as:

* baseline versus tool wear
* baseline versus misalignment
* baseline versus another simulated machine or process fault

The program performs exploratory statistical comparison. It does not claim to provide a trained or production-validated fault-classification model.

## Main capabilities

The software can:

* load referenced signal segments from MATLAB v7.3/HDF5 files
* analyse different input files through a command-line interface
* select an HDF5 group and sensor field
* compare a user-defined number of matching segments
* calculate several time-domain features
* handle missing or non-finite samples using a documented policy
* provide controlled errors for missing files, groups, fields and segments
* run without downloading the full dataset during testing

## Available signal features

The following features can be selected using the `--feature` argument:

| Feature            | CLI name       | Description                                                |
| ------------------ | -------------- | ---------------------------------------------------------- |
| Mean               | `mean`         | Arithmetic mean of the signal                              |
| Standard deviation | `std`          | Variation of signal samples                                |
| Minimum            | `min`          | Minimum recorded value                                     |
| Maximum            | `max`          | Maximum recorded value                                     |
| Root mean square   | `rms`          | Effective signal magnitude                                 |
| Peak-to-peak       | `peak_to_peak` | Difference between maximum and minimum                     |
| Kurtosis           | `kurtosis`     | Pearson kurtosis calculated from the fourth central moment |
| Crest factor       | `crest_factor` | Maximum absolute value divided by RMS                      |

The default comparison feature is standard deviation.

## Missing-data policy

Non-finite samples include:

* `NaN`
* positive infinity
* negative infinity

Two handling strategies are supported:

### Drop

```text
--missing-data drop
```

Non-finite samples are removed before features are calculated. This is the default behaviour.

### Raise

```text
--missing-data raise
```

The program stops and reports an error when a non-finite sample is found.

Signals containing no valid numeric samples are always rejected.

## Requirements

* Python 3.10 or later
* h5py
* NumPy
* pytest, for testing

## Installation

Clone the repository:

```bash
git clone https://github.com/abiswas2-stack/machining-tool-health-analysis.git
cd machining-tool-health-analysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Command-line usage

Run the software from the project root:

```bash
python src/main.py --file1 data/Segmented_Machining_Baseline.mat --group1 BaselineCrop --file2 data/Segmented_Machining_ToolWear.mat --group2 ToolWearCrop --field SpindleX --segments 10 --feature std --missing-data drop
```

### Required arguments

| Argument   | Description                                       |
| ---------- | ------------------------------------------------- |
| `--file1`  | Path to the baseline or first `.mat` file         |
| `--group1` | Top-level HDF5 group inside the first file        |
| `--file2`  | Path to the fault-condition or second `.mat` file |
| `--group2` | Top-level HDF5 group inside the second file       |

### Optional arguments

| Argument         |    Default | Description                                  |
| ---------------- | ---------: | -------------------------------------------- |
| `--field`        | `SpindleX` | Sensor field to analyse                      |
| `--segments`     |       `10` | Number of matching segments to compare       |
| `--feature`      |      `std` | Statistical feature used in the comparison   |
| `--missing-data` |     `drop` | Whether to drop or reject non-finite samples |

Display the help page with:

```bash
python src/main.py --help
```

## Example output

```text
Comparing 'data/Segmented_Machining_Baseline.mat' with 'data/Segmented_Machining_ToolWear.mat' for field 'SpindleX'
Feature: std
Segments compared: 10
Mean feature value (file1): 0.2148
Mean feature value (file2): 0.3512
Segments where file2 value is higher than file1: 80.0%
```

The numerical values above are illustrative. Actual output depends on the selected files, field, feature and number of segments.

## Running the tests

Run all tests from the repository root:

```bash
python -m pytest -q
```

The tests create small synthetic MATLAB-style HDF5 files at runtime. Therefore, the original machining dataset is not required to validate the software.

The test suite checks:

* signal loading
* statistical feature calculations
* segment counting
* baseline-versus-fault comparison
* invalid segment requests
* missing-data handling
* successful CLI execution
* CLI error handling

Tests are also executed automatically through GitHub Actions after pushes and pull requests.

## Project structure

```text
machining-tool-health-analysis/
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/
│   └── .mat files stored locally and not tracked
├── src/
│   ├── machining_file.py
│   ├── main.py
│   └── explore_data.py
├── tests/
│   ├── test_machining_file.py
│   └── test_cli.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Software design

The core logic is contained in the `MachiningFile` class.

Its responsibilities include:

* validating input files
* validating HDF5 groups and sensor fields
* loading individual signal segments
* cleaning missing data
* calculating time-domain features
* comparing features across matching segments

The command-line interface is kept separately in `src/main.py`. This separation prevents interface code from becoming mixed with data-loading and analysis logic.

## Validation approach

The software is validated using unit and integration tests based on synthetically generated signals with known values.

For example, the comparison test constructs:

* baseline signals with relatively low variability
* fault-condition signals with greater variability

The test then confirms that the calculated standard deviation is greater for the simulated fault condition.

Synthetic files reproduce the HDF5 reference structure expected by the loader while remaining small enough to store and run during automated testing.

## Fault classification (Random Forest)

In addition to single-feature comparisons, `src/fault_classifier.py` trains a Random Forest classifier using all eight computed features together, to distinguish between multiple machining conditions at once.

Run it directly with:

    python src/fault_classifier.py

On the four Machining conditions (Baseline, Misalignment, SurfaceCracks, ToolWear), this achieved approximately 91% overall accuracy. Baseline and ToolWear were classified almost perfectly, while Misalignment and SurfaceCracks were more frequently confused with each other, consistent with both being process/geometry-related faults rather than tool-condition faults.

The train/test split is stratified by condition but does not hold out entire files. Since segments from the same file originate from the same recording session, some session-specific effects could optimistically inflate accuracy slightly; a stricter evaluation would hold out whole files rather than individual segments.

## Limitations

The exploratory statistical comparisons (single-feature, segment-by-segment) assume that corresponding segments across two files represent comparable operations, and require the selected number of segments to exist in both files.

The Random Forest classifier improves on single-feature comparison by combining multiple features, but has its own limitations:

- the train/test split is stratified by condition but does not hold out entire files; since segments from the same file share a recording session, this may optimistically inflate accuracy slightly compared to evaluating on a fully unseen file
- the classifier was trained only on the four Machining conditions (Baseline, Misalignment, SurfaceCracks, ToolWear) and has not been evaluated on the Linear or Spindle fingerprint-routine conditions
- no threshold or confidence calibration has been validated for production deployment
- only time-domain statistical features are currently used; frequency-domain features (e.g. FFT-based) are not yet included and may further improve separation between Misalignment and SurfaceCracks, which the current model still confuses with each other

Future development could include frequency-domain features, evaluation with file-level (rather than segment-level) train/test splits, and validation on additional held-out experimental sessions.

## Licence

The source code in this repository is licensed under the MIT License. See the `LICENSE` file for details.

The machining dataset is not included in the software licence. Users must follow the terms provided by the original dataset publisher.
