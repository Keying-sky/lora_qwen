# M2 Coursework
### Keying Song (ks2146)

This coursework explores Low-Rank Adaptation (LoRA) for fine-tuning the Qwen2.5-Instruct Large Language Model (LLM) to predict multivariate time series data under tight compute budgets.

## Declaration
No auto-generation tools were used in this coursework.

## Project Structure
The main structure of the package `lora_qwen` is like:
```
.
├── lora_qwen/
│   ├── __init__.py               # expose all classes and functions for importing
│   ├── evaluate.py               # module for model evaluation and visualising some examples
│   ├── flops.py                  # module for FLOPS calculation
│   ├── preprocessor.py           # module for load and preprocess the data
│   ├── qwen.py                   # module for load Qwen model
│   └── train_lora.py             # module for model training with LoRA
|
├── report/                       # coursework report
|
├── data/                         # the dataset used
├── results/                      # the folder to save all the results
|
├── run/                          # the folder to directly answer the questions of coursework 
|
├── pyproject.toml                     
└── README.md               
```

## Installation

1. Clone the repository:
```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/m2_coursework/ks2146.git
```

2. Install: In the root directory:
```bash
pip install .
```

3. Use:
- After installing, all the classes and functions in `lora_qwen` can be imported and used anywhere on your own machine.
```python
from lora_qwen import evaluate_model, calculate_flops, train_lora_model, preprocess_trajectory
```

- Run the notebook files in the folder `run` to run all the experiments reported on the report.

## Usage

The main workflow runs intuitively in the folder `run` , in which the `run_baseline.ipynb` and `run_lora.ipynb` answer the questions in part 2 and 3 of the coursework respectively while the `run_flops_calculation.ipynb` demonstrates the calculation of the FLOPS used.
