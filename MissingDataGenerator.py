import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class MissingDataGenerator:
    def __init__(self, csv_path, missing_ratio):
        """
        Initialize the missing data generator
        
        Args:
            csv_path: Path to the original synthetic EHR dataset
            missing_ratio: Proportion of data to make missing (default 0.2 or 20%)
        """
        self.df = pd.read_csv(csv_path)
        self.missing_ratio = missing_ratio
        
        # Columns that can have missing values
        self.numeric_columns = [
            'temperature', 'heart_rate', 'blood_pressure_systolic',
            'blood_pressure_diastolic', 'respiratory_rate', 'oxygen_saturation',
            'wbc_count', 'rbc_count', 'hemoglobin', 'hematocrit',
            'platelet_count', 'sodium', 'potassium', 'chloride', 'glucose'
        ]
        
    def generate_MCAR(self):
        """
        Generate Missing Completely at Random (MCAR) dataset
        Missing values are introduced completely randomly
        """
        df_mcar = self.df.copy()
        
        for col in self.numeric_columns:
            # Randomly select indices to make missing
            mask = np.random.random(len(df_mcar)) < self.missing_ratio
            df_mcar.loc[mask, col] = np.nan
            
        return df_mcar
    
    def generate_MAR(self):
        """
        Generate Missing at Random (MAR) dataset
        Missing values depend on other observed variables
        """
        df_mar = self.df.copy()
        
        # Make lab values more likely to be missing if age is higher
        age_scaled = StandardScaler().fit_transform(df_mar[['age_at_visit']])
        
        for col in ['wbc_count', 'rbc_count', 'hemoglobin', 'hematocrit',
                   'platelet_count', 'sodium', 'potassium', 'chloride', 'glucose']:
            # Probability of missing increases with age
            missing_prob = self.missing_ratio * (1 + age_scaled.flatten())
            missing_prob = (missing_prob - missing_prob.min()) / (missing_prob.max() - missing_prob.min())
            
            # Generate missing values based on probabilities
            mask = np.random.random(len(df_mar)) < missing_prob
            df_mar.loc[mask, col] = np.nan
        
        # Make vital signs more likely to be missing if patient has fewer diagnoses
        diagnoses_count = df_mar.groupby('patient_id')['code'].transform('count')
        diagnoses_scaled = StandardScaler().fit_transform(diagnoses_count.values.reshape(-1, 1))
        
        for col in ['temperature', 'heart_rate', 'blood_pressure_systolic',
                   'blood_pressure_diastolic', 'respiratory_rate', 'oxygen_saturation']:
            # Probability of missing increases with fewer diagnoses
            missing_prob = self.missing_ratio * (1 - diagnoses_scaled.flatten())
            missing_prob = (missing_prob - missing_prob.min()) / (missing_prob.max() - missing_prob.min())
            
            mask = np.random.random(len(df_mar)) < missing_prob
            df_mar.loc[mask, col] = np.nan
            
        return df_mar
    
    def generate_MNAR(self):
        """
        Generate Missing Not at Random (MNAR) dataset
        Missing values depend on the values themselves
        """
        df_mnar = self.df.copy()
        
        # Higher values more likely to be missing
        for col in self.numeric_columns:
            values_scaled = StandardScaler().fit_transform(df_mnar[[col]])
            
            # Probability of missing increases with value
            missing_prob = self.missing_ratio * (1 + values_scaled.flatten())
            missing_prob = (missing_prob - missing_prob.min()) / (missing_prob.max() - missing_prob.min())
            
            # Generate missing values based on probabilities
            mask = np.random.random(len(df_mnar)) < missing_prob
            df_mnar.loc[mask, col] = np.nan
            
        return df_mnar
    
    def generate_combined(self):
        """
        Generate a dataset with a combination of MCAR, MAR, and MNAR missing values
        """
        df_combined = self.df.copy()
        
        # Split the missing ratio equally for MCAR, MAR, and MNAR
        mcar_ratio = self.missing_ratio / 3
        mar_ratio = self.missing_ratio / 3
        mnar_ratio = self.missing_ratio / 3
        
        # Apply MCAR mechanism
        for col in self.numeric_columns:
            mcar_mask = np.random.random(len(df_combined)) < mcar_ratio
            df_combined.loc[mcar_mask, col] = np.nan
            
        # Apply MAR mechanism
        age_scaled = StandardScaler().fit_transform(df_combined[['age_at_visit']])
        
        for col in ['wbc_count', 'rbc_count', 'hemoglobin', 'hematocrit',
                    'platelet_count', 'sodium', 'potassium', 'chloride', 'glucose']:
            mar_prob = mar_ratio * (1 + age_scaled.flatten())
            mar_prob = (mar_prob - mar_prob.min()) / (mar_prob.max() - mar_prob.min())
            mar_mask = np.random.random(len(df_combined)) < mar_prob
            df_combined.loc[mar_mask, col] = np.nan
        
        diagnoses_count = df_combined.groupby('patient_id')['code'].transform('count')
        diagnoses_scaled = StandardScaler().fit_transform(diagnoses_count.values.reshape(-1, 1))
        
        for col in ['temperature', 'heart_rate', 'blood_pressure_systolic',
                    'blood_pressure_diastolic', 'respiratory_rate', 'oxygen_saturation']:
            mar_prob = mar_ratio * (1 - diagnoses_scaled.flatten())
            mar_prob = (mar_prob - mar_prob.min()) / (mar_prob.max() - mar_prob.min())
            mar_mask = np.random.random(len(df_combined)) < mar_prob
            df_combined.loc[mar_mask, col] = np.nan
        
        # Apply MNAR mechanism
        for col in self.numeric_columns:
            values_scaled = StandardScaler().fit_transform(df_combined[[col]])
            mnar_prob = mnar_ratio * (1 + values_scaled.flatten())
            mnar_prob = (mnar_prob - mnar_prob.min()) / (mnar_prob.max() - mnar_prob.min())
            mnar_mask = np.random.random(len(df_combined)) < mnar_prob
            df_combined.loc[mnar_mask, col] = np.nan
        
        return df_combined
    
    def generate_all_datasets(self):
        """Generate all three types of missing datasets"""
        print("Generating MCAR dataset...")
        df_mcar = self.generate_MCAR()
        
        print("Generating MAR dataset...")
        df_mar = self.generate_MAR()
        
        print("Generating MNAR dataset...")
        df_mnar = self.generate_MNAR()
        
        print("Generating COMBINED dataset...")
        df_combined = self.generate_combined()
        
        return df_mcar, df_mar, df_mnar, df_combined

def generate_missing_datasets(input_path, missing_ratio):
    """
    Generate three datasets with different missing data mechanisms
    
    Args:
        input_path: Path to the original synthetic EHR dataset
        missing_ratio: Proportion of data to make missing
    """
    # Initialize generator
    generator = MissingDataGenerator(input_path, missing_ratio)
    
    # Generate datasets
    df_mcar, df_mar, df_mnar, df_combined = generator.generate_all_datasets()
    
    # Save datasets
    output_paths = []
    for df, mechanism in zip([df_mcar, df_mar, df_mnar, df_combined], ['MCAR', 'MAR', 'MNAR', 'COMBINED']):
        output_path = input_path.replace('.csv', f'_{mechanism}.csv')
        df.to_csv(output_path, index=False)
        output_paths.append(output_path)
        print(f"Saved {mechanism} dataset to {output_path}")
        
        # Print missing value statistics
        missing_counts = df.isnull().sum()
        print(f"\n{mechanism} Missing Value Statistics:")
        print(missing_counts[missing_counts > 0])
        
    return output_paths

if __name__ == "__main__":
    input_file = 'synthetic_ehr_dataset_train.csv'
    output_files = generate_missing_datasets(input_file, missing_ratio=0.2)