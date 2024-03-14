# Organoid Segmentation Using Self-Supervised Learning: How Complex Should the Pretext Task Be?
Code repository for the conference paper ["Organoid Segmentation Using Self-Supervised Learning: How Complex Should the Pretext Task Be?"](https://doi.org/10.1145/3637732.3637772) published and presented at the [Internation Conference on Biomedical and Bioinformatics Engineering (ICBBE)](https://www.icbbe.com/) [2023](https://dl.acm.org/doi/proceedings/10.1145/3637732).

NOTE: This repository is a work in progress.

## Setup
The usage of a virtual environment is recommended. To install all dependencies, run:
```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Usage
As described, training is divided into pretext training and downstream training.
### Pretext
Pretext training can be done using:
```
python pretext_train.py --task=[task option]
```
Here, `[task option]` consists of `{b,d,s,r,B,D,S,R}`.

Testing can be done using:
```
python pretext_test.py
```

### Downstream
Downstream training can be done using:
```
python downstream_train.py --task=[task option]
```

Downstream testing can be done using:
```
python downstream_test.py --task=[task option]
```

## Unit tests
The folder `tests/` contains python file(s) that unit test the modules from `src/`. The tests can be run using:
```
pytest tests/*.py
```
