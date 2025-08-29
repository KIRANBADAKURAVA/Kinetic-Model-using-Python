"""
Simple Random Forest Test Script
This script demonstrates Random Forest with a small sample of your kinetic data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

def create_sample_data():
    """
    Create sample data for testing Random Forest
    """
    print("Creating sample kinetic data for testing...")
    
    # Generate sample data similar to your kinetic data structure
    np.random.seed(42)
    n_samples = 1000
    
    # Temperature range (similar to your data)
    temperature = np.random.uniform(100, 1300, n_samples)
    
    # Time range
    time = np.random.uniform(0, 120, n_samples)
    
    # Mass percentage (decreasing with temperature)
    mass_percent = 100 - (temperature - 100) * 0.05 + np.random.normal(0, 0.5, n_samples)
    mass_percent = np.clip(mass_percent, 0, 100)
    
    # Alpha (conversion degree) - increasing with temperature
    alpha = (temperature - 100) / 1200 + np.random.normal(0, 0.05, n_samples)
    alpha = np.clip(alpha, 0, 1)
    
    # Heating rates
    heating_rates = np.random.choice([10, 20, 30], n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        '##Temp./°C': temperature,
        'Time/min': time,
        'Mass/%': mass_percent,
        'Alpha': alpha,
        'heating_rate': heating_rates,
        'Segment': np.ones(n_samples)
    })
    
    # Add some derived features
    data['temp_squared'] = data['##Temp./°C'] ** 2
    data['temp_inverse'] = 1 / (data['##Temp./°C'] + 273.15)
    data['mass_loss'] = 100 - data['Mass/%']
    data['alpha_squared'] = data['Alpha'] ** 2
    
    print(f"Sample data created: {data.shape[0]} samples, {data.shape[1]} features")
    return data

def test_random_forest_basic():
    """
    Basic Random Forest test
    """
    print("\n" + "="*50)
    print("BASIC RANDOM FOREST TEST")
    print("="*50)
    
    # Create sample data
    data = create_sample_data()
    
    # Prepare features
    feature_columns = ['##Temp./°C', 'Time/min', 'Mass/%', 'heating_rate', 
                      'temp_squared', 'temp_inverse', 'mass_loss', 'alpha_squared']
    
    X = data[feature_columns]
    y = data['Alpha']  # Predict alpha values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = rf_model.predict(X_train_scaled)
    y_pred_test = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Most Important Features:")
    print(importance_df.head())
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Alpha Values')
    plt.ylabel('Predicted Alpha Values')
    plt.title('Random Forest: Predicted vs Actual Alpha Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return rf_model, scaler, importance_df

def test_with_real_data():
    """
    Test with your actual data if available
    """
    print("\n" + "="*50)
    print("TESTING WITH REAL DATA")
    print("="*50)
    
    # Try to load your actual data
    possible_paths = [
        "data/Filterd Data/A2_filtered/HP0_10_A2_filtered.csv",
        "data/Filterd Data/D2_Filtered/HP0_10_D2_filtered.csv",
        "data/Filterd Data/F0_filterd/HP0_10_F0_filtered.csv"
    ]
    
    data_loaded = False
    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"Loading data from: {path}")
                data = pd.read_csv(path)
                
                # Prepare features
                feature_columns = ['##Temp./°C', 'Time/min', 'Mass/%', 'Alpha']
                if 'F_alpha' in data.columns:
                    feature_columns.append('F_alpha')
                if 'G_alpha' in data.columns:
                    feature_columns.append('G_alpha')
                
                # Remove rows with missing values
                data_clean = data[feature_columns].dropna()
                
                if len(data_clean) > 50:  # Ensure we have enough data
                    X = data_clean.drop('Alpha', axis=1)
                    y = data_clean['Alpha']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train model
                    print("Training Random Forest with real data...")
                    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                    rf_model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred_test = rf_model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    
                    print(f"Test R²: {test_r2:.4f}")
                    print(f"Test RMSE: {test_rmse:.4f}")
                    
                    # Feature importance
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': rf_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    print(f"\nFeature Importance with Real Data:")
                    print(importance_df)
                    
                    data_loaded = True
                    break
                    
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
    
    if not data_loaded:
        print("No real data files found. Using sample data only.")

def main():
    """
    Main execution function
    """
    print("Random Forest Test Script for Kinetic Analysis")
    print("=" * 60)
    
    # Test 1: Basic Random Forest with sample data
    rf_model, scaler, importance_df = test_random_forest_basic()
    
    # Test 2: Try with real data if available
    test_with_real_data()
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the feature importance plots")
    print("2. Check the prediction accuracy")
    print("3. Try with your own data files")
    print("4. Explore the advanced implementation in RandomForest_KineticAnalysis.py")

if __name__ == "__main__":
    main()
