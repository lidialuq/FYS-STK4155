## Project Description

The aim of this project was to compare common regression methods with the MLP Feedforward Neural Network when applied to a regression problem, as well as to compare logistic regression method with the MLP Feedforward Neural Network when applied to a classification problem. See the report for detailed explanation of the scope of this project. 

## Prerequisites

**Installing Python**

Make sure you have Python3 installed on your machine. This program is confirmed to work on version 3.8.10.

You may check your Python version by running:
```bash
python3 --version
```

**Installing dependencies**

Install all dependencies that are required for the project by running:
```bash
pip install -r requirements.txt
```

## Code Structure
```
README.md
requirements.txt
code/
|- data/
	|- optimization_sgd_files/
	|- breast_cancer.py
	|- franke_function.py
|- helpers/
	|- activations.py
	|- cost_functions.py
|- analysis.py
|- analysis_proj1.py
|- ffnn.py
|- experiments.py
|- main.py
|- optimize_sgd_params.py
|- sgd.py
figures/
|- breast_cancer/
|- franke_function/
```

## Usage
**Data**
data/franke_function.py: Samples, splits and normalises data from the Franke Function
data/breast_cancer.py: Access Wisconsin Breast Cancer Dataset, splits and normalises data.

**Models**
ffnn.py: Our Feed Forward Neural Network class. It uses helpers/activations.py and helpers/cost_functions.py as dependecies.
Fredrik sine

**Experiments**
experiments_ff_nn.py: All experiments with our FFNN using the Franke function dataset
experiments_bc_nn.py: All experiments with our FFNN using the Wisconsin Breast Cancer dataset


