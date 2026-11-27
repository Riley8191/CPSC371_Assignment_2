# CPSC371 Assignment 2 - Mushroom Classification

## Overview
Two classifiers implemented from scratch (no sklearn/ML libraries) to predict whether a mushroom is edible (`e`) or poisonous (`p`):
    1. **KNN** — `knn_classifier.py`
    2. **Perceptron** — `perceptron_classifier.py`

## Requirements
- Python 3.7+
- Standard library only (no external ML packages)

## Files
| File | Description |
|------|-------------|
| `knn_classifier.py` | KNN classifier + prediction script |
| `perceptron_classifier.py` | Perceptron classifier + prediction script |
| `MushroomData_8000.txt` | Training dataset (8000 instances) |
| `MushroomData_Unknwon_100.txt` | Unknown dataset (100 instances) |
| `predictionResultKNN.txt` | KNN predictions for the 100 unknowns |
| `predictionResultPER.txt` | Perceptron predictions for the 100 unknowns |
| `report.md` | Short report (settings, accuracy, comparisons) |

## How to Run

Place `MushroomData_8000.txt` and `MushroomData_Unknwon_100.txt` in the **same directory** as the scripts, then:

```bash
python3 knn_classifier.py          # generates predictionResultKNN.txt
python3 perceptron_classifier.py   # generates predictionResultPER.txt