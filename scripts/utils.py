import statsmodels.api as sm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os, re
from functools import reduce
from scipy.stats import zscore



# Redefining the compute_icc function
# Modified function to dynamically adjust for the number of raters
def compute_icc(data):
    """
    Computes the Intraclass Correlation Coefficient (ICC) using a two-way random effects model.
    Automatically adjusts for the number of raters.
    """
    # Determine the number of raters dynamically
    num_raters = data.shape[1]  # Number of columns = number of raters

    # Reshape data into long format
    melted_data = data.melt(ignore_index=False).reset_index()
    melted_data.columns = ["Speech", "Rater", "Score"]

    # Two-way random effects ANOVA model
    model = sm.formula.ols("Score ~ C(Speech) + C(Rater)", data=melted_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Extract mean squares
    MS_speech = anova_table["sum_sq"]["C(Speech)"] / anova_table["df"]["C(Speech)"]
    MS_error = anova_table["sum_sq"]["Residual"] / anova_table["df"]["Residual"]

    # Compute ICC(2,1) dynamically based on the number of raters
    ICC = (MS_speech - MS_error) / (MS_speech + (num_raters / num_raters) * MS_error)
    return ICC

def filter_columns(df, model, prompt):
    pattern = fr"{model}.*{prompt}"
    return df.filter(regex=pattern)

# Function to extract slice values and sort columns
def sort_columns_by_slice(df):
    def extract_slice(col):
        match = re.search(r"slice#([0-9.]+)", col)  # Extract numeric slice value
        return float(match.group(1)) if match else float("inf")  # Default to high value if missing

    sorted_columns = sorted(df.columns, key=lambda col: (extract_slice(col) != 1.0, extract_slice(col)))
    
    return df[sorted_columns]  # Reorder DataFrame columns

