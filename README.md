# BIOCLITE-PUCP: Transfer Learning for Bradykinesia Detection

## 📖 Description
This project implements transfer learning from supervised clinical settings to unsupervised home settings for bradykinesia detection in Parkinson's disease patients using smartwatch data.

## 👥 Dataset
- 40 participants (24 PD, 16 Healthy)
- 8 MDS-UPDRS exercises
- 50 Hz sampling rate
- Clinical (supervised) + Home (unsupervised) contexts

## 🏆 Best Model
**Random Forest** with engineered features:
- F1-score: 0.847 ± 0.069
- AUC: 0.884 ± 0.099
- Exercise 6 (Foot tapping)

## 📁 Project Structure
BIOCLITE-PUCP/
├── data/ # Datasets
│ ├── raw/ # Original CSV files
│ ├── processed/ # Preprocessed data
│ └── external/ # External datasets
├── src/ # Source code
│ ├── data/ # Data loading & preprocessing
│ ├── features/ # Feature extraction
│ ├── models/ # ML/DL models
│ ├── transfer/ # Transfer learning
│ ├── evaluation/ # Metrics & validation
│ ├── visualization/ # Plotting utilities
│ ├── xai/ # SHAP, LIME explainability
│ ├── experiments/ # Experiment scripts
│ └── publication/ # Paper figures & tables
├── notebooks/ # Jupyter notebooks
├── models/ # Saved trained models
├── results/ # Results & logs
├── docs/ # Documentation
├── papers/ # Paper drafts
└── tests/ # Unit tests

text

## 🚀 Quick Start
```python
from src.data.loader import BIOCLITEDataset
from src.models.classical import RandomForestModel

# Load data
loader = BIOCLITEDataset('data/raw/BIOCLITE_data_v2.csv')
df = loader.load_data()

# Train model
model = RandomForestModel()
model.fit(X_train, y_train)
📊 Results
Model	F1-score	AUC	Robustness
Random Forest	0.847	0.884	High
LSTM (6s filtered)	0.857	0.500	Low
LSTM (1s filtered)	0.571	0.633	Very Low
📝 Publication
Results submitted to [Journal Name]

👨‍🔬 Authors
BIOCLITE Research Group - PUCP

📄 License
[Your License]
