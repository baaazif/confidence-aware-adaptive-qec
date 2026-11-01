# Confidence-Aware Adaptive Quantum Error Correction Using AI

A research project investigating whether decoder confidence, calibration, and change detection can help an AI-assisted quantum error correction system adapt to changing noise conditions.

The central question:

**When should a quantum error decoder be trusted, adapted, or replaced because the noise environment has changed?**

Rather than assuming one decoder works equally well under every condition, this project explores the relationship between decoder reliability, confidence, noise variation, and adaptation cost.

## Research Overview

The project uses a simulated 3-qubit repetition code to investigate bit-flip errors and compare different decoding strategies.

The research progresses through:

- Baseline quantum error correction experiments.
- Dataset generation and feature engineering.
- Reference decoder implementation.
- Machine-learning decoder training and selection.
- Independent test-set evaluation.
- Evaluation under noise distribution shifts.
- History-aware decoding.
- Confidence calibration and risk analysis.
- Non-stationary noise detection using sliding windows and CUSUM.

## Implemented Components

### Quantum Error Correction
- 3-qubit repetition code simulation.
- Syndrome extraction and correction.
- Bit-flip noise experiments.
- Qiskit and Qiskit Aer simulation.

### Decoder Approaches
- Lookup-table decoder.
- Exact reference decoder.
- Machine-learning decoders using decision trees and logistic regression.
- History-aware minimum-weight decoding.

### Confidence and Adaptation
- Decoder confidence estimation.
- Probability calibration.
- Reliability and risk analysis.
- Sliding-window statistical monitoring.
- CUSUM-based change detection.

## Technology Stack

- **Language:** Python 3.12
- **Quantum simulation:** Qiskit 2.6.0, Qiskit Aer 0.17.2
- **Machine Learning:** scikit-learn
- **Testing:** pytest
- **Environment:** Python virtual environment

## Project Structure

```text
confidence-aware-qec/
├── docs/          # Research and experiment specifications
├── experiments/  # Reproducible experiment scripts
├── figures/      # Generated research figures
├── models/       # Trained model artifacts
├── results/      # Experimental outputs
├── src/caqec/    # Core project implementation
├── tests/        # Automated tests
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/baaazif/confidence-aware-adaptive-qec.git
cd confidence-aware-adaptive-qec
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it (Windows PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run tests

```bash
pytest
```

## Research Scope and Limitations

This is a simulation-based research project using a 3-qubit repetition code. It does not claim experimental validation on physical quantum hardware.

The work investigates decoder reliability and adaptation under simulated noise conditions. Results should be interpreted within this experimental scope.

## Research Direction

The broader goal is to explore confidence-aware decision-making for quantum error correction: determining not only which decoder performs best, but also when its predictions become unreliable and when adaptation is justified.

---

**Author:** [baaazif](https://github.com/baaazif)

[Portfolio](https://bazif.netlify.app) · [LinkedIn](https://www.linkedin.com/in/baaazif)
