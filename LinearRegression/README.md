# LinearRegression
Linear Regression Python implementation

## Use

Assure if your PC had all of these libs:

pandas / seaborn / matplotlib / numpy / scipy / scikit-learn / python-tk / python 2.7 or superior

### To compile:

On LinearRegression folder:

python -m py_compile methods/*.py model/*.py reader/*.py main.py

### To execute:

python main.py

**usage:** main.py [-h] [-model MODEL_TYPE] [-training TRAINING_PATH]
               [-test TEST_PATH] [-plot-data [PLOT_DATA]]
               [-plot-error [PLOT_ERROR]]

Linear Regression.

  **-h, --help**            show this help message and exit

  **-model (VOLUME | NOMINAL | TWOFEATURES | FOURFEATURES | DEFAULT)** - Chosen volume to avail

  **-training (TRAINING_PATH)** - the path of training set CSV file

  **-test (TEST_PATH)** - the path of test set CSV file

  **-plot-data [True|False]** - plot data visualization and data correlation

  **-plot-error [True|False]** - plot training error and correlation error on Gradient Descent implementations

