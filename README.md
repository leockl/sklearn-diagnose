# sklearn-diagnose

**An intelligent diagnosis layer for scikit-learn: evidence-based model failure detection with LLM-powered summaries.**

[![PyPI version](https://img.shields.io/pypi/v/sklearn-diagnose.svg)](https://pypi.org/project/sklearn-diagnose/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Philosophy

> **This library uses LLM-powered analysis for model diagnosis. All hypotheses are probabilistic and evidence-based.**

sklearn-diagnose acts as an "MRI scanner" for your machine learning models — it diagnoses problems but never modifies your models. The library follows an **evidence-first, LLM-powered** approach:

1. **Signal Extractors**: Compute deterministic statistics from your model and data
2. **LLM Hypothesis Generation**: Detect failure modes with confidence scores and severity
3. **LLM Recommendation Generation**: Generate actionable recommendations based on detected issues
4. **LLM Summary Generation**: Create human-readable summaries

## Key Features

- **Model Failure Diagnosis**: Detect overfitting, underfitting, high variance, label noise, feature redundancy, class imbalance, and data leakage symptoms
- **Cross-Validation Interpretation**: CV interpretation is a core signal extractor within sklearn-diagnose, used to detect instability, overfitting, and potential data leakage
- **Evidence-Based Hypotheses**: All diagnoses include confidence scores and supporting evidence
- **Actionable Recommendations**: Get specific suggestions to fix identified issues
- **Read-Only Behavior**: Never modifies your estimator, parameters, or data
- **Universal Compatibility**: Works with any fitted scikit-learn estimator or Pipeline

## Installation

```bash
pip install sklearn-diagnose
```

This installs sklearn-diagnose with all required dependencies including:
- **LangChain** (v1.2.0+) for AI agent capabilities
- **langchain-openai** for OpenAI model support
- **langchain-anthropic** for Anthropic model support
- **python-dotenv** for environment variable management

## Quick Start

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn_diagnose import setup_llm, diagnose

# Set up LLM (REQUIRED - must specify provider, model, and api_key)
# Using OpenAI:
setup_llm(provider="openai", model="gpt-4o", api_key="your-openai-key")
# setup_llm(provider="openai", model="gpt-4o-mini", api_key="your-openai-key")

# Or using Anthropic:
# setup_llm(provider="anthropic", model="claude-3-5-sonnet-latest", api_key="your-anthropic-key")

# Or using OpenRouter (access to many models):
# setup_llm(provider="openrouter", model="deepseek/deepseek-r1-0528", api_key="your-openrouter-key")

# Your existing sklearn workflow
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Diagnose your model
report = diagnose(
    estimator=model,
    datasets={
        "train": (X_train, y_train),
        "val": (X_val, y_val)
    },
    task="classification"
)

# View results
print(report.summary())          # LLM-generated summary
print(report.hypotheses)         # Detected issues with confidence
print(report.recommendations)    # LLM-ranked actionable suggestions
```

### With a Pipeline

```python
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn_diagnose import setup_llm, diagnose

# Set up LLM (required - do this once at startup)
os.environ["OPENAI_API_KEY"] = "your-key"
setup_llm(provider="openai", model="gpt-4o")  # api_key optional when env var set

# Build your pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
])

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression())
])
pipeline.fit(X_train, y_train)

# Diagnose works with any estimator
report = diagnose(
    estimator=pipeline,
    datasets={
        "train": (X_train, y_train),
        "val": (X_val, y_val)
    },
    task="classification"
)
```

### With Cross-Validation Results

```python
from sklearn.model_selection import cross_validate

# Run cross-validation
cv_results = cross_validate(
    model, X_train, y_train,
    cv=5,
    return_train_score=True,
    scoring='accuracy'
)

# Diagnose with CV evidence (no holdout set needed)
report = diagnose(
    estimator=model,
    datasets={
        "train": (X_train, y_train)
    },
    task="classification",
    cv_results=cv_results
)
```

## Detected Failure Modes

| Failure Mode | What It Detects | Key Signals |
|--------------|-----------------|-------------|
| **Overfitting** | Model memorizes training data | High train score, low val score, large gap |
| **Underfitting** | Model too simple for data | Low train and val scores |
| **High Variance** | Unstable across data splits | High CV fold variance, inconsistent predictions |
| **Label Noise** | Incorrect/noisy target labels | Ceilinged train score, scattered residuals |
| **Feature Redundancy** | Correlated/duplicate features | Detailed correlated pair list with correlation values |
| **Class Imbalance** | Skewed class distribution | Class distribution, per-class recall/precision, recall disparity |
| **Data Leakage** | Information from future/val in train | CV-to-holdout gap, suspicious feature-target correlations |

## Output Format

```python
report = diagnose(...)

# Human-readable summary (includes both diagnosis and recommendations)
report.summary()
# "## Diagnosis
#  Based on the analysis, here are the key findings:
#  - **Overfitting** (95% confidence, high severity)
#    - Train-val gap of 25.3% indicates overfitting
#  - **Feature Redundancy** (90% confidence, high severity)
#    - 4 highly correlated feature pairs detected (max correlation: 99.9%)
#    - Correlated feature pairs:
#    -   - Feature 0 ↔ Feature 10: 99.9% correlation
#    -   - Feature 1 ↔ Feature 11: 99.8% correlation
#  
#  ## Recommendations
#  **1. Increase regularization strength**
#     Stronger regularization penalizes model complexity..."

# Structured hypotheses with confidence scores
report.hypotheses
# [
#   Hypothesis(name=FailureMode.OVERFITTING, confidence=0.85, 
#              evidence=['Train-val gap of 23.0% is severe'], severity='high'),
#   Hypothesis(name=FailureMode.FEATURE_REDUNDANCY, confidence=0.90,
#              evidence=['4 highly correlated pairs detected',
#                        'Correlated feature pairs:',
#                        '  - Feature 0 ↔ Feature 10: 99.9% correlation',
#                        '  - Feature 1 ↔ Feature 11: 99.8% correlation'],
#              severity='high')
# ]

# Access hypothesis details
h = report.hypotheses[0]
h.name.value        # 'overfitting' (string)
h.confidence        # 0.85
h.evidence          # ['Train-val gap of 23.0% is severe']
h.severity          # 'high'

# Actionable recommendations (Recommendation objects)
report.recommendations
# [
#   Recommendation(action='Increase regularization strength', 
#                  rationale='Stronger regularization penalizes...',
#                  related_hypothesis=FailureMode.OVERFITTING),
#   Recommendation(action='Reduce model complexity', 
#                  rationale='Simpler models generalize better...',
#                  related_hypothesis=FailureMode.OVERFITTING)
# ]

# Access recommendation details
r = report.recommendations[0]
r.action              # 'Increase regularization strength'
r.rationale           # 'Stronger regularization penalizes...'
r.related_hypothesis  # FailureMode.OVERFITTING

# Raw signals (Signals object with attribute access)
report.signals.train_score      # 0.94
report.signals.val_score        # 0.71
report.signals.cv_mean          # 0.73 (if CV provided)
report.signals.cv_std           # 0.12 (if CV provided)
report.signals.to_dict()        # Convert to dict for serialization
```

## Design Principles

### Evidence-Based Diagnosis

Every hypothesis is backed by quantitative evidence. The LLM analyzes deterministic signals and generates hypotheses with confidence scores

### Confidence Scoring & Guardrails

- All hypotheses include explicit confidence scores (0.0 - 1.0)
- "Insufficient evidence" responses when signals are ambiguous
- Uncertainty is communicated clearly, never hidden
- No model changes are suggested automatically

### Read-Only Guarantee

sklearn-diagnose **never**:
- Calls `.fit()` on your estimator
- Modifies estimator parameters
- Mutates your training data
- Refits or retrains models

### Validation Set vs Cross-Validation

sklearn-diagnose follows strict rules:

1. **`y_val` is OPTIONAL** — You can diagnose with only training data + CV results
2. **CV evidence overrides holdout logic** — When both present, CV provides richer signals
3. **Never mix the two** — Holdout and CV answer different questions

## API Reference

### `diagnose()`

Main entry point for model diagnosis.

```python
def diagnose(
    estimator,              # Any fitted sklearn estimator or Pipeline
    datasets: dict,         # {"train": (X, y), "val": (X, y)} - val is optional
    task: str,              # "classification" or "regression"
    cv_results: dict = None # Output from cross_validate() - optional
) -> DiagnosisReport:
```

**Parameters:**

- `estimator`: A fitted scikit-learn estimator or Pipeline. Must already be fitted.
- `datasets`: Dictionary with "train" key required, "val" key optional. Each value is a tuple of (X, y).
- `task`: Either "classification" or "regression"
- `cv_results`: Optional dictionary from `sklearn.model_selection.cross_validate()`

**Returns:**

`DiagnosisReport` object with:
- `.hypotheses`: List of detected issues with confidence scores
- `.recommendations`: List of actionable fix suggestions (LLM-ranked)
- `.signals`: Raw computed statistics
- `.summary()`: Human-readable summary (LLM-generated)

## Configuration

### LLM Backend (Required)

sklearn-diagnose uses **LangChain** under the hood for LLM integration. Each diagnosis involves three AI agents:
- **Hypothesis Agent**: Analyzes signals and detects failure modes
- **Recommendation Agent**: Generates actionable fix suggestions  
- **Summary Agent**: Creates human-readable summaries

```python
from sklearn_diagnose import setup_llm

# Using OpenAI
setup_llm(provider="openai", model="gpt-4o", api_key="sk-...")

# Using Anthropic
setup_llm(provider="anthropic", model="claude-3-5-sonnet-latest", api_key="sk-ant-...")

# Using OpenRouter (access to many models)
setup_llm(provider="openrouter", model="deepseek/deepseek-r1-0528", api_key="sk-or-...")
```

### Using Environment Variables

You can set API keys via environment variables in two ways:

**Option 1: Set programmatically in Python**

```python
import os
from sklearn_diagnose import setup_llm

# Set environment variable in your code
os.environ["OPENAI_API_KEY"] = "sk-..."
setup_llm(provider="openai", model="gpt-4o")  # api_key is automatically loaded

# Or for Anthropic
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
setup_llm(provider="anthropic", model="claude-3-5-sonnet-latest")

# Or for OpenRouter
os.environ["OPENROUTER_API_KEY"] = "sk-or-..."
setup_llm(provider="openrouter", model="deepseek/deepseek-r1-0528")
```

**Option 2: Use a `.env` file (recommended for production)**

Create a `.env` file in your project root:

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

The library uses `python-dotenv` internally to automatically load the `.env` file (no need to import or call load_dotenv() yourself):

```python
from sklearn_diagnose import setup_llm

# API keys are automatically loaded from .env file
setup_llm(provider="openai", model="gpt-4o")
setup_llm(provider="anthropic", model="claude-3-5-sonnet-latest")
setup_llm(provider="openrouter", model="deepseek/deepseek-r1-0528")
```

## Architecture

```
sklearn-diagnose/                 # Project root
├── sklearn_diagnose/             # Main package
│   ├── __init__.py               # Package exports (setup_llm, diagnose, types)
│   ├── api/
│   │   ├── __init__.py
│   │   └── diagnose.py           # Main diagnose() function
│   ├── core/
│   │   ├── __init__.py
│   │   ├── schemas.py            # Data structures (Evidence, Signals, Hypothesis, etc.)
│   │   ├── evidence.py           # Input validation, read-only guarantees
│   │   ├── signals.py            # Signal extraction (deterministic metrics)
│   │   ├── hypotheses.py         # Rule-based hypotheses (fallback/reference)
│   │   └── recommendations.py    # Example recommendation templates for LLM guidance
│   └── llm/
│       ├── __init__.py           # Exports setup_llm and LLM utilities
│       └── client.py             # LangChain-based AI agents (hypothesis, recommendation, summary)
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures and MockLLMClient for testing
│   └── test_diagnose.py          # Comprehensive test suite
├── .github/
│   └── workflows/
│       └── tests.yml             # GitHub Actions CI (runs tests on push/PR)
├── .env.example                  # Template for API keys (copy to .env)
├── .gitignore
├── AGENTS.md                     # AI agents architecture documentation
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── MANIFEST.in
├── README.md
└── pyproject.toml
```

### Processing Flow

```
User Input (model, data, task)
         │
         ▼
┌─────────────────────────────┐
│  1. Signal Extraction       │  Deterministic metrics
│     (signals.py)            │  (train_score, val_score, cv_std, etc.)
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  2. Hypothesis Agent        │  Failure modes with confidence & severity
│     (LangChain create_agent)│  (overfitting: 95%, high severity)
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  3. Recommendation Agent    │  Actionable recommendations
│     (LangChain create_agent)│  (guided by example templates)
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  4. Summary Agent           │  Human-readable summary
│     (LangChain create_agent)│
└─────────────────────────────┘
         │
         ▼
    DiagnosisReport
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use sklearn-diagnose in your research, please cite:

```bibtex
@software{sklearn_diagnose,
  title = {sklearn-diagnose: Evidence-based model failure diagnosis for scikit-learn},
  year = {2025},
  url = {https://github.com/leockl/sklearn-diagnose}
}
```

---

Please give my GitHub repo a ⭐ if this was helpful. Thank you! 🙏
