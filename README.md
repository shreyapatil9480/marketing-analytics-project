# Marketing Analytics Project

This project demonstrates core skills in **business analytics**, **data analytics**, and **marketing analytics** using a synthetic dataset that emulates typical marketing campaign data. The goal is to showcase proficiency in data wrangling, exploratory analysis, customer segmentation, predictive modeling, and visualization. The dataset is **generated on the fly** by the analysis script, so no external data files are required.

## Project Overview

Marketing teams often seek to understand customer behavior, segment customers into actionable groups, and predict which customers are most likely to respond to a campaign. In this project you will find:

* **Data generation:** A synthetic dataset with 1 000 customer records is automatically created by `analysis.py` if it does not find an existing CSV. Each record contains demographic information (age, income, gender, marital status), behavioral metrics (recency, frequency, monetary value), tenure as a customer, and a binary **Response** variable indicating whether the customer responded to a marketing offer.
* **Exploratory analysis:** Summary statistics and visualizations for key features. The script produces histograms of age by gender and boxplots of spend by marital status (saved in the `plots/` folder).
* **Customer segmentation:** K‑means clustering on frequency and monetary value to identify customer segments. A scatter plot of the clusters is saved as `customer_segments.png`.
* **Predictive modeling:** A logistic regression model predicts the **Response** variable using demographic, behavioral and cluster information. The model is evaluated with cross‑validation and a classification report on a hold‑out test set.

## Repository Structure

```
marketing_analytics_project/
├── analysis.py          # Main script performing data generation, EDA, clustering and modeling
├── requirements.txt     # Python dependencies
├── README.md            # Project overview and instructions
└── .gitignore           # Ignore generated data and plots
```

## Getting Started

Follow these steps to replicate the analysis on your local machine:

1. **Clone or download this repository** from GitHub.
2. **Create a virtual environment** (recommended) and install the dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the analysis script:**

   ```bash
   python analysis.py
   ```

   When executed, the script will generate a synthetic dataset (if one does not already exist), create plots in a `plots/` folder, perform customer segmentation, train a logistic regression model, and print evaluation metrics to the console.

## Interpreting the Results

The synthetic dataset is designed such that higher income, higher purchase frequency, and recent activity increase the likelihood of a customer responding to a campaign. After running the script you should see:

* **Cluster visualization:** Customers are grouped into four segments based on frequency and monetary value. Segments can be interpreted as high‑value frequent buyers, occasional big spenders, low‑value infrequent buyers, etc.
* **Model performance:** The cross‑validated accuracy scores provide a baseline measure for the logistic regression model. A classification report on a hold‑out test set shows precision, recall and F1‑score for the positive (responding) and negative classes.

## Notes

* The dataset is synthetically generated for demonstration purposes. In a real‑world project you would replace this with an actual marketing or e‑commerce dataset.
* To extend the analysis, consider experimenting with different clustering algorithms (e.g., hierarchical clustering), additional features (e.g., channel preference), or more advanced predictive models (e.g., random forests, gradient boosting).

## License

This repository is provided for educational purposes. You are free to modify and use the code as needed.