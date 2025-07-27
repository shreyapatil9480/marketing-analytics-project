"""
analysis.py
================

This script performs exploratory data analysis, customer segmentation, and response prediction on a
synthetic marketing dataset. The dataset is generated to resemble common characteristics in
marketing analytics, including customer demographics (age, income, gender, marital status),
behavioral metrics (recency, frequency, monetary value), and a binary response indicating
whether a customer responded to a marketing campaign.

The analysis includes:

* Loading the data from `synthetic_marketing_data.csv`.
* Summary statistics and distribution plots for key features.
* K‑means clustering on frequency and monetary variables to identify customer segments.
* Logistic regression to predict campaign response based on customer features.
* Evaluation of the predictive model using cross‑validation.

To run this script, install the required packages (see requirements.txt) and execute:

    python analysis.py

This will produce a `plots` directory containing PNG files of the generated charts and
print evaluation metrics to stdout.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


def load_data(path: str) -> pd.DataFrame:
    """Load the synthetic marketing dataset from a CSV file."""
    df = pd.read_csv(path)
    # Ensure correct dtypes
    df['Gender'] = df['Gender'].astype('category')
    df['Marital_Status'] = df['Marital_Status'].astype('category')
    return df


def summarize_data(df: pd.DataFrame) -> None:
    """Print summary statistics and save distribution plots."""
    print("Summary statistics:\n")
    print(df.describe(include='all'))

    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    # Histogram of Age by Gender
    for gender in df['Gender'].cat.categories:
        subset = df[df['Gender'] == gender]
        plt.hist(subset['Age'], bins=15, alpha=0.5, label=gender)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Age Distribution by Gender')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'age_distribution_by_gender.png')
    plt.close()

    # Boxplot of Monetary spend by Marital Status
    df.boxplot(column='Monetary', by='Marital_Status', figsize=(8, 6))
    plt.title('Monetary Spend by Marital Status')
    plt.suptitle('')
    plt.ylabel('Monetary Spend ($)')
    plt.tight_layout()
    plt.savefig(plots_dir / 'monetary_by_marital_status.png')
    plt.close()


def perform_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform K‑means clustering on Frequency and Monetary features and return the dataframe with cluster labels."""
    features = df[['Frequency', 'Monetary']].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Choose 4 clusters (common in marketing segmentation)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    df['Cluster'] = clusters

    # Plot clusters
    plt.figure(figsize=(8, 6))
    for cluster_id in np.unique(clusters):
        cluster_points = scaled_features[clusters == cluster_id]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}', alpha=0.6
        )
    plt.xlabel('Frequency (scaled)')
    plt.ylabel('Monetary (scaled)')
    plt.title('Customer Segments based on Frequency and Monetary')
    plt.legend()
    plt.tight_layout()
    plots_dir = Path('plots')
    plt.savefig(plots_dir / 'customer_segments.png')
    plt.close()
    return df


def train_model(df: pd.DataFrame) -> None:
    """Train a logistic regression model to predict Response and evaluate it using cross‑validation."""
    X = df.drop(columns=['Response'])
    y = df['Response']

    # Preprocess: numeric features scaled, categorical encoded
    numeric_features = ['Age', 'Income', 'Tenure_Months', 'Recency', 'Frequency', 'Monetary']
    categorical_features = ['Gender', 'Marital_Status', 'Cluster']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000)),
        ]
    )

    # Evaluate using 5‑fold cross‑validation
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"Cross‑validated accuracy scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Fit on full data and print classification report on a hold‑out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\nClassification report on hold‑out test set:")
    print(classification_report(y_test, y_pred))


def generate_synthetic_dataset(path: str, n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic marketing dataset and save it to `path`.

    This helper replicates the data generation process used during development.
    It is invoked automatically if no CSV is found at `path`.
    """
    rng = np.random.default_rng(random_state)
    ages = rng.integers(18, 70, size=n_samples)
    income = rng.normal(loc=60000, scale=20000, size=n_samples)
    income = np.clip(income, 10000, None)
    gender = rng.choice(['Male', 'Female'], size=n_samples)
    marital_status = rng.choice(['Single', 'Married', 'Divorced', 'Widowed'], size=n_samples)
    tenure_months = rng.integers(1, 121, size=n_samples)
    recency = rng.integers(0, 100, size=n_samples)
    frequency = rng.integers(1, 20, size=n_samples)
    monetary = rng.normal(loc=2000, scale=500, size=n_samples) + frequency * 50
    monetary = np.clip(monetary, 0, None)
    gender_indicator = (gender == 'Female').astype(int)
    marital_indicator = (marital_status == 'Married').astype(int)
    score = -3 + (income / 100000) * 3 + frequency * 0.1 - recency * 0.02 + gender_indicator * 0.5 + marital_indicator * 0.3
    prob = 1 / (1 + np.exp(-score))
    response = rng.binomial(1, prob)
    df = pd.DataFrame({
        'Age': ages,
        'Income': np.round(income, 2),
        'Gender': gender,
        'Marital_Status': marital_status,
        'Tenure_Months': tenure_months,
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': np.round(monetary, 2),
        'Response': response,
    })
    df.to_csv(path, index=False)
    return df


def main() -> None:
    """Main entry point for the analysis script."""
    data_path = 'synthetic_marketing_data.csv'
    if not Path(data_path).is_file():
        print(f"Dataset '{data_path}' not found. Generating a synthetic dataset...")
        df = generate_synthetic_dataset(data_path)
    else:
        df = load_data(data_path)
    summarize_data(df)
    df_clustered = perform_clustering(df)
    train_model(df_clustered)


if __name__ == '__main__':
    main()