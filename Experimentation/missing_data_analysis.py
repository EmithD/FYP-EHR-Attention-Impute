import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def analyze_missing_values(file_path):
    """
    Comprehensive analysis of missing values in the dataset
    """
    # Read the dataset
    print(f"Reading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Basic missing value information
    print("\n1. BASIC MISSING VALUE SUMMARY:")
    print("-" * 50)
    total_cells = np.prod(df.shape)
    total_missing = df.isnull().sum().sum()
    
    print(f"Total number of cells in dataset: {total_cells:,}")
    print(f"Total number of missing values: {total_missing:,}")
    print(f"Percentage of missing values: {(total_missing/total_cells)*100:.2f}%")
    
    # 2. Missing values by column
    print("\n2. MISSING VALUES BY COLUMN:")
    print("-" * 50)
    missing_by_column = df.isnull().sum()
    missing_by_column_pct = (df.isnull().sum() / len(df)) * 100
    
    missing_table = PrettyTable()
    missing_table.field_names = ["Column", "Missing Count", "Missing Percentage"]
    
    for col in df.columns:
        if missing_by_column[col] > 0:
            missing_table.add_row([
                col, 
                missing_by_column[col], 
                f"{missing_by_column_pct[col]:.2f}%"
            ])
    
    if missing_table._rows:
        print(missing_table)
    else:
        print("No missing values found in any column!")
    
    # 3. Generate missing value heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Value Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.tight_layout()
    plt.savefig('missing_values_heatmap.png')
    plt.close()
    
    # 4. Check for patterns in missing values
    if total_missing > 0:
        print("\n3. MISSING VALUE PATTERNS:")
        print("-" * 50)
        
        # Check for rows with multiple missing values
        rows_with_missing = df.isnull().sum(axis=1)
        print(f"Rows with at least one missing value: {(rows_with_missing > 0).sum():,}")
        print("\nDistribution of missing values per row:")
        print(rows_with_missing[rows_with_missing > 0].value_counts().sort_index())
        
        # Correlation between missing values in different columns
        columns_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]
        if len(columns_with_missing) > 1:
            print("\nCorrelation between missing values in different columns:")
            missing_correlations = df[columns_with_missing].isnull().corr()
            print(missing_correlations)
    
    # 5. Summary statistics for numeric columns with missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols_with_missing = [col for col in numeric_cols if df[col].isnull().sum() > 0]
    
    if numeric_cols_with_missing:
        print("\n4. STATISTICS FOR NUMERIC COLUMNS WITH MISSING VALUES:")
        print("-" * 50)
        stats_table = PrettyTable()
        stats_table.field_names = ["Column", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        
        for col in numeric_cols_with_missing:
            stats = df[col].describe()
            stats_table.add_row([
                col,
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['25%']:.2f}",
                f"{stats['50%']:.2f}",
                f"{stats['75%']:.2f}",
                f"{stats['max']:.2f}"
            ])
        
        print(stats_table)
        
if __name__ == "__main__":
    # Analyze the original synthetic dataset
    file_path = 'synthetic_ehr_dataset_train.csv'
    analyze_missing_values(file_path)

    print("\nNote: A heatmap visualization has been saved as 'missing_values_heatmap.png'")