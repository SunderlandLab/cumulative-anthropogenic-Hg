## Introduction
This repository contains source code for the paper:

> Geyman, B.M., D.G. Streets, C.I. Olson, C.P. Thackray, C.L. Olson, K. Schaefer, D.P. Krabbenhoft, and E.M. Sunderland (2025). Cumulative Anthropogenic Impacts of Past and Future Releases on the Global Mercury Cycle. *Environmental Science & Technology*

Anthropogenic emissions/releases and compiled seawater mercury data used in the analysis are stored separately on <a href="https://doi.org/10.7910/DVN/CWX5PO">Harvard Dataverse</a> and can be downloaded to the repository directory using `fetch_data.sh`. Code to support box modeling is located in a separate GitHub <a href="https://github.com/SunderlandLab/boxey">Repository</a> and can be installed through the method outlined below.

The script `install_environment.sh` creates a conda environment called `pyboxey_env` that installs python and all dependencies needed to run the code. The script `call_all.sh` executes the complete analysis in sequence.

__________

### Installation and Usage Instructions

### Mac
 - Install miniconda [<a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html">Link</a>]
 - Download and unzip `cumulative-anthropogenic-Hg` repository
 - Open the terminal. At the terminal prompt, type:
   ```
   cd ~/Downloads/cumulative-anthropogenic-Hg/scripts/ # it might alternately be ~/Downloads/cumulative-anthropogenic-Hg-main/scripts/
   bash install_environment.sh
   conda activate pyboxey_env
   bash call_all.sh
   ```


