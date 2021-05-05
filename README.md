# SBFC

Seed-based resting state functional connectivity with Nilearn.

This project is subject to change as Nilearn GLM features are still under development.

[![Test and coverage](https://github.com/htwangtw/sbfc/actions/workflows/main.yml/badge.svg)](https://github.com/htwangtw/sbfc/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/htwangtw/sbfc/branch/main/graph/badge.svg?token=014wbcr2sJ)](https://codecov.io/gh/htwangtw/sbfc)

## Dependencies

The required dependencies to use the software are:

* Python >= 3.7
* Nilearn >= 0.7.0
* Matplotlib >= 3.4.0

## Install

First make sure you have installed all the dependencies listed above.
Then you can install by running the following command in
a command prompt:

```bash
pip install git+https://github.com/htwangtw/sbfc.git
```

## Prepare your data

This library work on minimally processed data only.
If you need to preprocess your imaging data, please consider `fMRIprep`.

You can find an example in `example` and files that you should prepare to run the pipeline.
