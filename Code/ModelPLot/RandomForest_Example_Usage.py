"""
Random Forest Example Usage for Kinetic Analysis
This script demonstrates how to integrate Random Forest with your existing kinetic data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
import glob

def load_multiple_datasets(base_path):
    """
    Load multiple datasets from your filtered data directory
    """
    all_data = []
    
    # Get all CSV files from the filtered data directories
    csv_files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)
    
    for file_path in csv_files:
        try:
            # Extract condition and model info from filename
            filename = os.path.basename(file_path)
            parts = filename.replace('.csv', '').split('_')
            
            # Parse condition and heating rate
            condition = parts[0]  # HP0, HP1, etc.
            heating_rate = parts[1]  # 10, 20, 30
            model_type = parts[2]  # A2, D2, F0, etc.
            
            # Load data
            data = pd.read_csv(file_path)
            
            # Add metadata columns
            data['condition'] = condition
            data['heating_rate'] = int(heating_rate)
            data['model_type'] = model_type
            data['file_source'] = filename
            
            all_data.append(data)
            print(f"Loaded: {filename} - {len(data)} rows")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined dataset: {combined_data.shape[0]} rows, {combined_data.shape[1]} columns")
        return combined_data
    else:
        raise ValueError("No data files found!")

def prepare_features_for_rf(data):
    """
    Prepare features for Random Forest analysis
    """
    features = pd.DataFrame()
    
    # Basic experimental features
    features['temperature'] = data['##Temp./°C']
    features['time'] = data['Time/min']
    features['mass_percent'] = data['Mass/%']
    features['alpha'] = data['Alpha']
    
    # Derived features
    features['temp_squared'] = data['##Temp./°C'] ** 2
    features['temp_inverse'] = 1 / (data['##Temp./°C'] + 273.15)  # Kelvin
    features['mass_loss'] = 100 - data['Mass/%']
    features['alpha_squared'] = data['Alpha'] ** 2
    
    # Kinetic function features
    if 'F_alpha' in data.columns:
        features['f_alpha'] = data['F_alpha']
    if 'G_alpha' in data.columns:
        features['g_alpha'] = data['G_alpha']
    
    # Experimental condition features
    features['heating_rate'] = data['heating_rate']
    features['segment'] = data['Segment']
    
    # Encode categorical variables
    features['condition_encoded'] = pd.Categorical(data['condition']).codes
    features['model_type_encoded'] = pd.Categorical(data['model_type']).codes
    
    # Interaction features
    features['temp_alpha_interaction'] = features['temperature'] * features['alpha']
    features['temp_heating_rate_interaction'] = features['temperature'] * features['heating_rate']
    features['alpha_heating_rate_interaction'] = features['alpha'] * features['heating_rate']
    
    return features

def train_rf_regression_model(features, target, test_size=0.2):
    """
    Train Random Forest regression model
    """
    print("Training Random Forest Regression Model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
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
    
    return {
        'model': rf_model,
        'scaler': scaler,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
        'metrics': {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_rmse': train_rmse, 'test_rmse': test_rmse
        }
    }

def analyze_feature_importance(model, feature_names):
    """
    Analyze and visualize feature importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Random Forest Feature Importance', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    return importance_df

def plot_predictions_vs_actual(y_true, y_pred, title):
    """
    Plot predictions vs actual values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def cross_validation_analysis(features, target, cv_folds=5):
    """
    Perform cross-validation analysis
    """
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(rf_model, features_scaled, target, cv=cv_folds, scoring='r2')
    
    print(f"Cross-validation R² scores: {scores}")
    print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return scores

def main():
    """
    Main execution function
    """
    print("Random Forest Kinetic Analysis Example")
    print("=" * 50)
    
    # Load data from your filtered data directory
    base_path = "data/Filterd Data"
    
    try:
        # Load combined dataset
        combined_data = load_multiple_datasets(base_path)
        
        # Prepare features
        features = prepare_features_for_rf(combined_data)
        
        # Remove any rows with missing values
        features = features.dropna()
        combined_data = combined_data.loc[features.index]
        
        print(f"\nFinal dataset: {features.shape[0]} samples, {features.shape[1]} features")
        
        # Choose target variable (you can modify this based on your needs)
        target_options = {
            'alpha': combined_data['Alpha'],
            'temperature': combined_data['##Temp./°C'],
            'mass_percent': combined_data['Mass/%']
        }
        
        # For this example, let's predict alpha values
        target = target_options['alpha']
        
        print(f"\nTarget variable: Alpha (conversion degree)")
        print(f"Target range: {target.min():.3f} to {target.max():.3f}")
        
        # Perform cross-validation
        cv_scores = cross_validation_analysis(features, target)
        
        # Train Random Forest model
        results = train_rf_regression_model(features, target)
        
        # Analyze feature importance
        importance_df = analyze_feature_importance(results['model'], features.columns)
        
        # Plot predictions vs actual
        plot_predictions_vs_actual(
            results['y_test'], 
            results['y_pred_test'], 
            'Random Forest: Predicted vs Actual Alpha Values'
        )
        
        # Save results
        results_df = pd.DataFrame({
            'actual': results['y_test'],
            'predicted': results['y_pred_test']
        })
        results_df.to_csv('rf_predictions_results.csv', index=False)
        print(f"\nResults saved to: rf_predictions_results.csv")
        
        # Summary
        print(f"\n{'='*50}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Dataset size: {features.shape[0]} samples")
        print(f"Number of features: {features.shape[1]}")
        print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test R²: {results['metrics']['test_r2']:.4f}")
        print(f"Test RMSE: {results['metrics']['test_rmse']:.4f}")
        print(f"Top feature: {importance_df.iloc[0]['feature']} (importance: {importance_df.iloc[0]['importance']:.4f})")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure your data path is correct and data files exist.")

if __name__ == "__main__":
    main()
