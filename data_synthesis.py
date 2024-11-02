import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import random
from faker import Faker
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
# seed used for training dataset - 1
# seed used for testing dataset - 42

fake = Faker()

def generate_vital_signs():
    """Generate realistic vital signs"""
    return {
        'temperature': round(np.random.normal(37.0, 0.4), 1),
        'heart_rate': int(np.random.normal(75, 12)),
        'blood_pressure_systolic': int(np.random.normal(120, 10)),
        'blood_pressure_diastolic': int(np.random.normal(80, 8)),
        'respiratory_rate': int(np.random.normal(16, 2)),
        'oxygen_saturation': int(np.random.normal(97, 2))
    }
    
def generate_lab_results():
    """Generate synthetic lab results"""
    return {
        'wbc_count': round(np.random.normal(7.5, 2), 1),
        'rbc_count': round(np.random.normal(4.5, 0.5), 1),
        'hemoglobin': round(np.random.normal(14, 1.5), 1),
        'hematocrit': round(np.random.normal(42, 4), 1),
        'platelet_count': int(np.random.normal(250000, 50000)),
        'sodium': int(np.random.normal(140, 3)),
        'potassium': round(np.random.normal(4.0, 0.4), 1),
        'chloride': int(np.random.normal(102, 3)),
        'glucose': int(np.random.normal(100, 15))
    }
    
# ICD-10 codes for medical conditions
conditions = {
    'I10': 'Essential hypertension',
    'E11.9': 'Type 2 diabetes without complications',
    'J44.9': 'COPD',
    'I25.10': 'Coronary artery disease',
    'F41.1': 'Generalized anxiety disorder',
    'F32.9': 'Major depressive disorder',
    'M17.9': 'Osteoarthritis',
    'E78.5': 'Dyslipidemia',
    'N18.3': 'Chronic kidney disease, stage 3',
    'G47.33': 'Obstructive sleep apnea'
}

medications = [
    'Lisinopril 10mg',
    'Metformin 500mg',
    'Atorvastatin 40mg',
    'Amlodipine 5mg',
    'Metoprolol 25mg',
    'Sertraline 50mg',
    'Omeprazole 20mg',
    'Levothyroxine 50mcg',
    'Hydrochlorothiazide 25mg',
    'Aspirin 81mg'
]

def generate_synthetic_ehr(num_patients):
    """Generate synthetic EHR dataset"""
    
    records = []
    
    print("Generating synthetic EHR records...")
    for _ in tqdm(range(num_patients)):
        # Generate basic patient information
        patient_id = fake.uuid4()
        dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
        gender = random.choice(['M', 'F'])
        
        # Generate multiple visits for each patient
        num_visits = random.randint(1, 10)
        
        for visit in range(num_visits):
            visit_date = fake.date_between(
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2024, 1, 1)
            )
            
            # Generate vital signs and lab results
            vitals = generate_vital_signs()
            labs = generate_lab_results()
            
            # Assign random conditions (1-4 conditions per visit)
            patient_conditions = random.sample(list(conditions.items()), 
                                            random.randint(1, 4))
            
            # Assign random medications (1-5 medications per visit)
            patient_medications = random.sample(medications, 
                                             random.randint(1, 5))
            
            # Create record
            record = {
                'patient_id': patient_id,
                'visit_id': fake.uuid4(),
                'visit_date': visit_date,
                'date_of_birth': dob,
                'gender': gender,
                'age_at_visit': (visit_date.year - dob.year - 
                               ((visit_date.month, visit_date.day) < 
                                (dob.month, dob.day))),
                'diagnoses': [{'code': code, 'description': desc} 
                            for code, desc in patient_conditions],
                'medications': patient_medications,
                **vitals,
                **labs
            }
            
            records.append(record)
    
    # Convert to DataFrame
    df = pd.json_normalize(records, 
                         sep='_',
                         record_path=['diagnoses'],
                         meta=['patient_id', 'visit_id', 'visit_date', 
                               'date_of_birth', 'gender', 'age_at_visit',
                               'medications',
                               'temperature', 'heart_rate', 
                               'blood_pressure_systolic',
                               'blood_pressure_diastolic',
                               'respiratory_rate', 'oxygen_saturation',
                               'wbc_count', 'rbc_count', 'hemoglobin',
                               'hematocrit', 'platelet_count', 'sodium',
                               'potassium', 'chloride', 'glucose'])
    
    return df

dataset_size = 1000
df = generate_synthetic_ehr(dataset_size)

df.to_csv('synthetic_ehr_dataset_test.csv', index=False)
print(f"Generated {len(df)} records for {dataset_size} patients")

# Basic statistics
print("\nDataset Statistics:")
print(f"Total number of visits: {len(df)}")
print(f"Unique patients: {df['patient_id'].nunique()}")
print(f"Average visits per patient: {len(df)/df['patient_id'].nunique():.2f}")
print(f"Date range: {df['visit_date'].min()} to {df['visit_date'].max()}")