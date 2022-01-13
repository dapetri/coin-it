# coin-it

RL model levareging on correlations between crypto coins.

# Setup

The project consists of multiple stages, named:

- dataset
- training
- ...

All those stages have their own [conda](https://anaconda.org) environment,
to reduce the size of each environment and prevent dependency conflicts
(especially with different python versions).

## Create Environment

All files are named _coin-it\_%STAGE%.yml_,
where _%STAGE%_ must be replaced with one of the stages listed above.
To ensure all notebooks execute correctly, the last command given below adds the PYTHONPATH
environment variable pointing to the src folder to the active conda environment.

To create an environment execute following commands

    conda env create -f ./envs/coin-it_%STAGE%.yml
    conda activate coin-it_%STAGE%
    conda env config vars set PYTHONPATH=$PWD/src

E.g. to generate the environment for _dataset_, execute:

    conda env create -f ./envs/coin-it_dataset.yml
    conda activate coin-it_training
    conda env config vars set PYTHONPATH=$PWD/src

## Update Environment

Make sure you are in the correct conda environment

    conda activate coin-it_%STAGE%

execute following commands to update conda and pip packages

    conda env export > ./envs/coin-it_%STAGE%.yml

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
