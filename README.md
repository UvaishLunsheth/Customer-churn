# Customer-churn

## 🐍 Conda Environment Setup

A dedicated Conda virtual environment is used to ensure dependency isolation and reproducibility.

### Environment Details
- Environment Name: `churn_env`
- Python Version: `3.9`

### Steps to Create Environment

```bash
conda create -n churn_env python=3.9 -y
conda activate churn_env

## 📦 Package Setup

The project is structured as an installable Python package using `setup.py`.
This allows clean imports and editable installation during development.

### Editable Install

```bash
pip install -e .
