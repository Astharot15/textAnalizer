# Transformer Model Explainability

## Overview
This project applies black-box (SHAP) and white-box (transformers-interpret) explanation methods to a transformer-based text classification model.

## Installation
```bash
pip install torch transformers shap captum transformers-interpret
```

## Usage
### Black-Box Explanation (SHAP)
Run:
```bash
python black_box.py
```
This generates an HTML file (`shap_explanation.html`) with SHAP-based explanations.

### White-Box Explanation (transformers-interpret)
Run:
```bash
python white_box.py
```
This generates an HTML file (`explanation_visualizations.html`) with gradient-based explanations.

## Results
Both methods provide insights into model decision-making. SHAP is model-agnostic, while transformers-interpret uses gradients for token-level attributions.
