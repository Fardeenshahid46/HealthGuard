import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def perform_basic_eda(df):
    print("First 5 rows of the dataset:",df.head())
    print("\nDataset Info:")
    print("\n",df.info())
    print("\nSummary statistics:")
    print("\n",df.describe())
    print("\nMissing values in each column:")
    print("\n",df.isnull().sum())
    print("\nCorrelation matrix:")
    print("\n",df.corr())

def clean_data(df):
    columns_with_zero=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_with_zero] = df[columns_with_zero].replace(0, np.nan)
    
    #Fill NaN with median values
    df.fillna(df.median(), inplace=True)
    
    #scale features
    scaler=StandardScaler()
    X=df.drop(columns=['Outcome'])
    X_scaled=scaler.fit_transform(X)
    df_scaled=pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['Outcome']=df['Outcome'].values
    
    return df_scaled

if __name__ == "__main__":
    os.makedirs("data",exist_ok=True)
    df=pd.read_csv("data/diabetes.csv")
    
    perform_basic_eda(df)
    cleaned_df=clean_data(df)
    
    cleaned_path="data/cleaned_diabetes.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"\n✅ Cleaned dataset saved to: {cleaned_path}")
    