# 🚢 Titanic Survival Prediction

A machine learning project on the Kaggle **Titanic – Machine Learning from Disaster** training data
(`titanic.csv`, 891 passengers). It covers data exploration with pandas/seaborn, a perceptron built
from scratch, and a **Keras neural network** (with Keras Tuner experiments) that predicts whether a
passenger survived. An interactive **Streamlit** app serves the trained network.

## 🚀 Features

- Exploratory analysis and pandas exercises on the Titanic data
- Perceptron with sigmoid activation and gradient descent implemented in NumPy
- Neural network classifier (`Deep_Learning.ipynb`):
  `Dense(128) → Dropout(0.5) → Dense(64) → Dense(64, L1) → Dropout(0.7) → Dense(32, L2) → Dropout(0.3) → Dense(1, sigmoid)`,
  SGD with momentum, early stopping, plus a Keras Tuner random search
- **Streamlit demo**:
  - Predict a passenger: survival probability, the actual survival rate of similar passengers, and a
    "what if" chart of probability vs. age
  - Batch prediction from a CSV in Kaggle format (missing values imputed), with download
  - Model performance: accuracy, precision, recall, F1, ROC AUC, confusion matrix, ROC curve, training history
  - Data insights: survival rate by sex, class, age group and port

## 📊 Model

| | |
|---|---|
| Features | `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked` |
| Preprocessing | Median imputation (Age, Fare), most-frequent (Embarked), ordinal encoding (Sex, Embarked), standard scaling — fitted on the training split |
| Split | 80/20, `random_state=42` (712 train / 179 test) |

Test-set results of the saved model (`models/`):

| Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|
| 0.816 | 0.815 | 0.716 | 0.763 | 0.880 |

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · TensorFlow / Keras · Keras Tuner · seaborn · Altair · Streamlit · Jupyter

## 📂 Project Structure

```
titanic/
├── titanic.csv                    # Kaggle Titanic training data
├── Deep_Learning.ipynb            # Neural network + Keras Tuner
├── titanic_data.ipynb             # Missing values / EDA
├── pandas.ipynb                   # pandas exercises
├── perceptron_from_scratch.ipynb  # NumPy perceptron
├── titanic_model.py               # Preprocessing, training, saving, prediction
├── app.py                         # Streamlit demo
├── models/
│   ├── titanic_mlp.keras          # Trained network
│   ├── preprocessor.joblib        # Fitted imputers, encoder and scaler
│   └── metadata.json              # Test metrics, ROC curve, training history
├── requirements.txt
└── .streamlit/config.toml
```

## 📦 Installation

```bash
git clone https://github.com/amirafarhan/titanic.git
cd titanic
python -m venv venv
# Windows: venv\Scripts\activate    Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Run

Streamlit demo (uses the saved model in `models/`):

```bash
streamlit run app.py
```

Re-train the model and regenerate `models/`:

```bash
python titanic_model.py
```

Notebooks need a few extra packages:

```bash
pip install jupyter seaborn keras-tuner tensorboard
```

Then run `jupyter notebook`.

## 📈 Insights from the data

- Women survived far more often than men (74% vs. 19%)
- Survival drops with ticket class (1st 63%, 2nd 47%, 3rd 24%)
- Children under 13 had a higher survival rate than other age groups

## ⚠️ Limitations

- Trained on 712 passengers; the test set has 179, so metrics vary a few points between splits.
- `Name`, `Ticket` and `Cabin` are not used (Cabin is missing for 77% of passengers).

## 🔮 Future Improvements

- Engineer features such as title (from `Name`), family size and deck (from `Cabin`)
- Cross-validated comparison with tree-based models
- Predict on the Kaggle test set and submit

## 👨‍💻 Author

Amira Farhan – Machine Learning & Computer Vision Engineer
