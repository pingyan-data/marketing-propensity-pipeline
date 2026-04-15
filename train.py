import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import MinMaxScaler, OneHotEncoder, OrdinalEncoder, Binarizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import (balanced_accuracy_score, recall_score,
                                       precision_score, roc_curve, auc)
from tqdm import tqdm
import os

os.makedirs("plots", exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────
df = pd.read_parquet("customer_data.parquet")
print(f"Loaded {len(df)} records. Target rate: {df['target'].mean():.1%}")

# ── 2. Define feature groups ───────────────────────────────────
NUMERICAL   = ["age", "tenure_months"]
BINARY      = ["gender", "has_car"]
ONEHOT      = ["region", "segment"]
ORDINAL     = ["income_index", "product_holdings",
               "digital_score", "activity_score",
               "engagement_level", "housing_type"]
TARGET      = "target"
DROP_COLS   = ["customer_id", TARGET]

# ── 3. Preprocessing pipelines ─────────────────────────────────
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler",  MinMaxScaler()),
])
onehot_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
ordinal_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder()),
])
binary_pipe = Pipeline([
    ("imputer",   SimpleImputer(strategy="most_frequent")),
    ("binarizer", Binarizer()),
])

preprocessor = ColumnTransformer([
    ("num",    num_pipe,     NUMERICAL),
    ("onehot", onehot_pipe,  ONEHOT),
    ("ord",    ordinal_pipe, ORDINAL),
    ("bin",    binary_pipe,  BINARY),
])

# Fit preprocessor once to get feature names
X_all = df.drop(columns=DROP_COLS)
y_all = df[TARGET]
preprocessor.fit(X_all)
features_out = preprocessor.get_feature_names_out()
print(f"Total features after preprocessing: {len(features_out)}")

# ── 4. Train/val split ──────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ── 5. Feature selection sweep with MLflow ─────────────────────
k_range     = range(5, len(features_out), 3)
best_auc    = 0
best_k      = None
results     = []

mlflow.set_experiment("propensity_model")

with mlflow.start_run(run_name="feature_selection_sweep"):
    for k in tqdm(k_range, desc="Sweeping k"):
        with mlflow.start_run(run_name=f"k={k}", nested=True):

            classifier = LogisticRegression(class_weight="balanced", max_iter=500)
            pipeline   = Pipeline([
                ("preprocessor",     preprocessor),
                ("feature_selection", SelectKBest(f_classif, k=k)),
                ("classifier",        classifier),
            ])

            pipeline.fit(X_train, y_train)

            y_val_pred       = pipeline.predict(X_val)
            y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]

            bal_acc   = balanced_accuracy_score(y_val, y_val_pred)
            recall    = recall_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred)
            fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
            roc_auc   = auc(fpr, tpr)

            mlflow.log_params({"k": k, "model": "LogisticRegression"})
            mlflow.log_metrics({
                "val_balanced_accuracy": bal_acc,
                "val_recall":            recall,
                "val_precision":         precision,
                "val_auc":               roc_auc,
            })

            results.append({"k": k, "auc": roc_auc,
                             "recall": recall, "precision": precision})

            if roc_auc > best_auc:
                best_auc      = roc_auc
                best_k        = k
                best_pipeline = pipeline
                best_fpr, best_tpr = fpr, tpr

    # ── 6. Log best model ──────────────────────────────────────
    mlflow.log_params({"best_k": best_k})
    mlflow.log_metrics({"best_val_auc": best_auc})
    mlflow.sklearn.log_model(best_pipeline, "best_model")
    print(f"\nBest k={best_k} | AUC={best_auc:.3f}")

# ── 7. Save results & plots ────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("sweep_results.csv", index=False)

# Plot 1: AUC vs k
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(results_df["k"], results_df["auc"], color="#378ADD", lw=2, marker="o", markersize=3)
ax.axvline(best_k, color="#D85A30", linestyle="--", label=f"Best k={best_k}")
ax.set_xlabel("Number of features (k)")
ax.set_ylabel("Validation AUC")
ax.set_title("AUC vs Number of Selected Features")
ax.legend()
fig.tight_layout()
fig.savefig("plots/auc_vs_k.png", dpi=150)
print("Saved plots/auc_vs_k.png")

# Plot 2: ROC curve for best model
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.plot(best_fpr, best_tpr, color="#378ADD", lw=2,
         label=f"ROC (AUC = {best_auc:.2f})")
ax2.plot([0, 1], [0, 1], color="#888780", linestyle="--")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title(f"ROC Curve — Best Model (k={best_k})")
ax2.legend(loc="lower right")
fig2.tight_layout()
fig2.savefig("plots/roc_curve.png", dpi=150)
print("Saved plots/roc_curve.png")

# Plot 3: Score distribution
y_val_scores = best_pipeline.predict_proba(X_val)[:, 1]
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.hist(y_val_scores[y_val == 0], bins=40, alpha=0.6,
         color="#378ADD", label="Non-target")
ax3.hist(y_val_scores[y_val == 1], bins=40, alpha=0.6,
         color="#D85A30", label="Target")
ax3.set_xlabel("Predicted Propensity Score")
ax3.set_ylabel("Count")
ax3.set_title("Score Distribution by Class")
ax3.legend()
fig3.tight_layout()
fig3.savefig("plots/score_distribution.png", dpi=150)
print("Saved plots/score_distribution.png")

# Plot 4: Cumulative gains curve
sorted_idx   = np.argsort(-y_val_scores)
sorted_true  = np.array(y_val)[sorted_idx]
cum_gains    = np.cumsum(sorted_true) / sorted_true.sum()
population   = np.arange(1, len(sorted_true) + 1) / len(sorted_true)

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(population * 100, cum_gains * 100,
         color="#378ADD", lw=2, label="Model")
ax4.plot([0, 100], [0, 100], color="#888780",
         linestyle="--", label="Random baseline")
ax4.axvline(20, color="#D85A30", linestyle=":", alpha=0.7)
ax4.set_xlabel("% Population Contacted")
ax4.set_ylabel("% Targets Captured")
ax4.set_title("Cumulative Gains Curve")
ax4.legend()
fig4.tight_layout()
fig4.savefig("plots/cumulative_gains.png", dpi=150)
print("Saved plots/cumulative_gains.png")

print("\nAll done.")