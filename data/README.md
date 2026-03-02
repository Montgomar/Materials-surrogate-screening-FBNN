# Data format

This project reads training data from Excel files.

## Training Excel (example: `TrainingData-ANN-All.xlsx`)
- One row = one experimental sample
- Columns include:
  - input features (X): material composition, test conditions
  - targets (Y): COF, specific wear rate

Required:
- A header row with unique column names
- No merged cells

Note:
- Proprietary data is not committed to this repository.

[//]: # (- Provide a `data/demo/` dataset for reproducibility if possible.)