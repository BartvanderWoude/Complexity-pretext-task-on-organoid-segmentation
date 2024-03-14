# Organoid Segmentation Using Self-Supervised Learning: How Complex Should the Pretext Task Be?
Code repository for the conference paper ["Organoid Segmentation Using Self-Supervised Learning: How Complex Should the Pretext Task Be?"](https://doi.org/10.1145/3637732.3637772) published and presented at the [Internation Conference on Biomedical and Bioinformatics Engineering (ICBBE)](https://www.icbbe.com/) [2023](https://dl.acm.org/doi/proceedings/10.1145/3637732).

## Setup
The usage of a virtual environment is recommended. To install all dependencies, run:
```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Unit tests
The folder `tests/` contains python file(s) that unit test the modules from `src/`. The tests can be run using:
```
pytest tests/*.py
```
