#Akilan Final Project AI Present and Future


# ==================== ENVIRONMENT SETUP ====================
import os
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ==================== IMPORT LIBRARIES ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                           balanced_accuracy_score, fbeta_score, roc_auc_score,
                           precision_recall_curve, PrecisionRecallDisplay,
                           average_precision_score, roc_curve)
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                   StratifiedKFold, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.tree import plot_tree
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Input, BatchNormalization,
                                   Activation)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                      ReduceLROnPlateau)
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
import shap

# ==================== GLOBAL CONFIGURATION ====================
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create directories for outputs
os.makedirs("Plots", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("monitoring", exist_ok=True)

# ==================== ENHANCED FEATURE ENGINEERING ====================
def enhanced_feature_engineering(df):
    """Add more sophisticated feature engineering"""
    # Interaction features
    df['NexE'] = df['Nscore'] * df['Escore']
    df['OexC'] = df['Oscore'] * df['Cscore']

    # Risk score variants
    df['Risk_Score2'] = df['Impulsive'] + df['SS']
    df['Risk_Score3'] = np.sqrt(df['Impulsive']**2 + df['SS']**2)

    # Personality aggregates
    df['Personality_Std'] = df[['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']].std(axis=1)
    df['Personality_Range'] = df[['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']].max(axis=1) - df[['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']].min(axis=1)

    return df

# ==================== DATA LOADING & PREPROCESSING ====================
def load_and_preprocess_data(target='Alcohol'):
    """Enhanced data loading and preprocessing"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data"
    column_names = [
        'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
        'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
        'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc',
        'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD',
        'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
    ]

    print("Loading dataset...")
    df = pd.read_csv(url, header=None, names=column_names)

    # Convert drug use to binary (CL0=0, others=1)
    drug_columns = [target] + [
        'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc',
        'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD',
        'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
    ]

    for col in drug_columns:
        df[col] = df[col].str.extract(r'CL(\d+)')[0].astype(int)
        df[col] = (df[col] > 0).astype(int)

    # Enhanced feature engineering
    df = enhanced_feature_engineering(df)
    df['Personality_Score'] = df[['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']].mean(axis=1)
    df['Risk_Score'] = df['Impulsive'] * df['SS']
    df['Age_Group'] = pd.cut(df['Age'], bins=[18,25,35,50,100],
                            labels=['18-25','26-35','36-50','50+'])

    # Encode categorical features
    categorical_cols = ['Gender', 'Education', 'Country', 'Ethnicity', 'Age_Group']
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df, drug_columns

# ==================== IMPROVED FEATURE PREPARATION ====================
def prepare_features(df, target='Alcohol', drug_columns=None):
    """Enhanced feature preparation with robust feature selection"""
    if drug_columns is None:
        drug_columns = [
            'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc',
            'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD',
            'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
        ]

    X = df.drop(columns=[target, 'ID'] + [col for col in df.columns
                                        if col.startswith('CL') or col in drug_columns[1:]])
    y = df[target]

    print("\n=== Class Distribution ===")
    class_dist = y.value_counts(normalize=True)
    print(class_dist)

    # Scaling and feature selection are fitted later on the TRAINING split only
    # (see fit_feature_pipeline); fitting them here on all rows leaked test data.
    return X, y, class_dist


def fit_feature_pipeline(X_train, y_train, X_other_sets, class_dist):
    """Fit the scaler and XGBoost feature selector on the training split, then apply to the other splits."""
    X_train = X_train.copy()
    numerical_cols = X_train.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    transformed = []
    for X_o in X_other_sets:
        X_o = X_o.copy()
        X_o[numerical_cols] = scaler.transform(X_o[numerical_cols])
        transformed.append(X_o)

    print("\nPerforming feature selection (training split only)...")
    selector = SelectFromModel(
        XGBClassifier(
            scale_pos_weight=class_dist[1]/class_dist[0],
            random_state=SEED,
            eval_metric='aucpr'
        ),
        threshold='median'
    )
    X_train_sel = selector.fit_transform(X_train, y_train)
    selected_feature_names = X_train.columns[selector.get_support()]
    others_sel = [selector.transform(X_o) for X_o in transformed]
    joblib.dump((scaler, selector), 'models/feature_pipeline.pkl')
    return X_train_sel, others_sel, selected_feature_names

# ==================== IMPROVED MODEL TRAINING ====================
def train_decision_tree(X_train, y_train):
    """Train and optimize decision tree with cross-validation"""
    print("\n=== Training Decision Tree ===")
    params = {
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'ccp_alpha': [0.0, 0.01, 0.1]  # For pruning
    }

    model = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        param_distributions=params,
        n_iter=25,
        cv=StratifiedKFold(5),
        scoring='balanced_accuracy',
        n_jobs=-1,
        random_state=SEED
    )

    model.fit(X_train, y_train)
    joblib.dump(model.best_estimator_, 'models/best_dt_model.pkl')
    print(f"Best DT Parameters: {model.best_params_}")
    return model.best_estimator_

def train_xgboost(X_train, y_train, class_dist):
    """Enhanced XGBoost training with better hyperparameter tuning"""
    print("\n=== Training XGBoost ===")

    # Expanded parameter grid with more options
    params = {
        'scale_pos_weight': [class_dist[1]/class_dist[0]],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.001, 0.01, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1],
        'eval_metric': ['aucpr'],
        'random_state': [SEED]
    }

    # Using RandomizedSearchCV for better hyperparameter search
    model = RandomizedSearchCV(
        XGBClassifier(),
        param_distributions=params,
        n_iter=50,
        cv=StratifiedKFold(5),
        scoring='average_precision',
        n_jobs=1,  # Reduced to 1 to avoid potential issues
        verbose=1,
        random_state=SEED
    )

    model.fit(X_train, y_train)

    joblib.dump(model.best_estimator_, 'models/best_xgb_model.pkl')
    print(f"Best Parameters: {model.best_params_}")
    return model.best_estimator_

from sklearn.utils.class_weight import compute_class_weight

def train_neural_network(X_train, y_train, class_dist):
    """Optimized Neural Network with stratified validation, simplified architecture, and proper class weight handling"""
    print("\n=== Training Neural Network ===")

    # Manual stratified validation split
    X_nn_train, X_val, y_nn_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )

    # Compute class weights as a dictionary
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_nn_train), y=y_nn_train)
    class_weight_dict = dict(zip(np.unique(y_nn_train), class_weights))

    # Neural network model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            AUC(name='prc', curve='PR')
        ]
    )

    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, monitor='val_auc', mode='max'),
        ModelCheckpoint('models/best_nn_model.keras', save_best_only=True, monitor='val_auc', mode='max'),
        ReduceLROnPlateau(factor=0.2, patience=10, min_lr=1e-6)
    ]

    history = model.fit(
        X_nn_train, y_nn_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=128,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    return model, history

# ==================== ENHANCED EVALUATION ====================
def predict_proba_any(model, X, model_type):
    """Positive-class probabilities for the sklearn/XGBoost models and the Keras network."""
    if model_type in ('xgb', 'dt'):
        return model.predict_proba(X)[:, 1].astype(float)
    return model.predict(X, verbose=0).flatten().astype(float)


def choose_threshold(model, X_val, y_val, model_type):
    """Youden's J threshold chosen on the VALIDATION split, so the test set stays untouched."""
    fpr, tpr, thresholds = roc_curve(y_val, predict_proba_any(model, X_val, model_type))
    return float(thresholds[np.argmax(tpr - fpr)])


def evaluate_model(model, X_test, y_test, model_type, threshold=None):
    """Comprehensive model evaluation with optimal threshold finding"""
    if model_type == 'xgb':
        y_proba = model.predict_proba(X_test)[:, 1]
    elif model_type == 'dt':
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test).flatten()

    # Convert probabilities to Python native float type
    y_proba = y_proba.astype(float)

    # Find optimal threshold using Youden's J statistic.
    # Note: when no threshold is passed this tunes on the evaluation data itself (optimistic);
    # the main script passes a threshold chosen on the validation split instead.
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        threshold = float(thresholds[optimal_idx])  # Convert to Python float

    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics and ensure all values are JSON serializable
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f2_score': float(fbeta_score(y_test, y_pred, beta=2)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'pr_auc': float(average_precision_score(y_test, y_proba)),
        'threshold': float(threshold),
        'recall_at_90precision': float(recall_at_precision(y_test, y_proba, 0.9))
    }


    print(f"Test Accuracy: {metrics['accuracy']:.2%}")
    print(f"F2-Score: {metrics['f2_score']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"PR AUC: {metrics['pr_auc']:.3f}")
    print(f"Recall at 90% Precision: {metrics['recall_at_90precision']:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save metrics for monitoring
    with open(f"monitoring/{model_type}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.title(f'Precision-Recall Curve ({model_type.upper()})', fontsize=14)
    plt.grid(True)
    plt.savefig(f"Plots/pr_curve_{model_type}.png", bbox_inches='tight', dpi=300)
    plt.close()

    return y_pred, y_proba, metrics

def recall_at_precision(y_true, y_proba, precision_threshold):
    """Calculate recall at specified precision threshold"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return recall[np.searchsorted(precision[:-1], precision_threshold)]

def plot_enhanced_confusion_matrix(y_true, y_pred, model_type):
    """Enhanced confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Consumer', 'Consumer'],
                yticklabels=['Non-Consumer', 'Consumer'],
                cbar=False)
    plt.title(f'Confusion Matrix ({model_type.upper()})', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.savefig(f"Plots/confusion_matrix_{model_type}.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_training_history(history):
    """Enhanced training history plots"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    ax1.plot(history.history['prc'], label='Train')
    ax1.plot(history.history['val_prc'], label='Validation')
    ax1.set_title('Precision-Recall AUC', fontsize=12)
    ax1.set_ylabel('PR AUC', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    ax3.plot(history.history['recall'], label='Train Recall')
    ax3.plot(history.history['precision'], label='Train Precision')
    ax3.plot(history.history['val_recall'], label='Val Recall')
    ax3.plot(history.history['val_precision'], label='Val Precision')
    ax3.set_title('Precision & Recall', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("Plots/nn_training_history.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_decision_tree_structure(model, feature_names):
    """Plot the structure of the trained Decision Tree"""
    plt.figure(figsize=(24, 16))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Non-Consumer", "Consumer"],
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth = 3

    )
    plt.title("Decision Tree Structure (Heading)", fontsize=12)
    plt.savefig("Plots/decision_tree_structure.png", bbox_inches='tight', dpi=300)
    plt.close()

def explain_model(model, X_test, model_type, feature_names):
    """Generate SHAP explanations for model interpretability"""
    background = shap.maskers.Independent(X_test[:100], max_samples=100)
    explainer = shap.Explainer(model, background, feature_names=feature_names)
    shap_values = explainer(X_test[:100])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:100], show=False, feature_names=feature_names)
    plt.title(f'SHAP Summary ({model_type.upper()})', fontsize=14)
    plt.savefig(f"Plots/shap_{model_type}.png", bbox_inches='tight', dpi=300)
    plt.close()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Load and preprocess data
    df, drug_columns = load_and_preprocess_data()
    X, y, class_dist = prepare_features(df)

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    # Hold out part of the training data (before resampling) to choose decision thresholds
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train
    )

    # Fit scaling + feature selection on the training split only, then apply to val and test
    X_train, (X_val, X_test), feature_names = fit_feature_pipeline(X_train, y_train, [X_val, X_test], class_dist)

    # Robust resampling with fallback
    print("\nBalancing classes with resampling...")
    try:
        smt = SMOTETomek(sampling_strategy='auto', random_state=SEED, n_jobs=1)
        X_train, y_train = smt.fit_resample(X_train, y_train)
        print("Resampled class distribution (SMOTETomek):", pd.Series(y_train).value_counts())
    except Exception as e:
        print(f"SMOTETomek failed, using SMOTE instead. Error: {str(e)}")
        smt = SMOTE(sampling_strategy='auto', random_state=SEED)
        X_train, y_train = smt.fit_resample(X_train, y_train)
        print("Resampled class distribution (SMOTE):", pd.Series(y_train).value_counts())

    # Train and evaluate all models
    dt_model = train_decision_tree(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train, class_dist)
    nn_model, history = train_neural_network(X_train, y_train, class_dist)

    # Evaluate all models on the test set, with thresholds chosen on the validation split
    thr = {name: choose_threshold(m, X_val, y_val, name)
           for name, m in (('dt', dt_model), ('xgb', xgb_model), ('nn', nn_model))}
    y_pred_dt, y_proba_dt, dt_metrics = evaluate_model(dt_model, X_test, y_test, 'dt', threshold=thr['dt'])
    y_pred_xgb, y_proba_xgb, xgb_metrics = evaluate_model(xgb_model, X_test, y_test, 'xgb', threshold=thr['xgb'])
    y_pred_nn, y_proba_nn, nn_metrics = evaluate_model(nn_model, X_test, y_test, 'nn', threshold=thr['nn'])

    # Plot confusion matrices
    plot_enhanced_confusion_matrix(y_test, y_pred_dt, 'dt')
    plot_enhanced_confusion_matrix(y_test, y_pred_xgb, 'xgb')
    plot_enhanced_confusion_matrix(y_test, y_pred_nn, 'nn')

    # Plot the trained Decision Tree structure
    plot_decision_tree_structure(dt_model, feature_names)

    # SHAP explainability
    explain_model(xgb_model, X_test, "xgb", feature_names)
    explain_model(nn_model, X_test, "nn", feature_names)

    # Enhanced training history plots
    plot_training_history(history)

    # Model comparison
    print("\n=== Final Model Comparison ===")
    print(f"Decision Tree - Test Accuracy: {dt_metrics['accuracy']:.2%} ")
    print(f"XGBoost       - Test Accuracy: {xgb_metrics['accuracy']:.2%}")
    print(f"Neural Net    - Test Accuracy: {nn_metrics['accuracy']:.2%} ")
    print("\n=== Training Complete ===")
