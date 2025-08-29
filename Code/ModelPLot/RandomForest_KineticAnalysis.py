import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class KineticRandomForest:
    """
    Random Forest implementation for thermal kinetic analysis
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.regressor = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare_data(self, data_path, model_type='regression'):
        """
        Load and prepare data for Random Forest analysis
        
        Parameters:
        - data_path: Path to the data file
        - model_type: 'regression' for Ea prediction, 'classification' for material classification
        """
        print("Loading and preparing data...")
        
        # Load data
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'):
            data = pd.read_excel(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
        
        # Extract features
        self.features = self._extract_features(data)
        self.target = self._extract_target(data, model_type)
        
        # Handle missing values
        self.features = self.features.fillna(self.features.mean())
        
        # Encode categorical variables
        self.features = self._encode_categorical_features(self.features)
        
        print(f"Data prepared: {self.features.shape[0]} samples, {self.features.shape[1]} features")
        return self.features, self.target
    
    def _extract_features(self, data):
        """Extract relevant features from kinetic data"""
        features = pd.DataFrame()
        
        # Basic experimental features
        if '##Temp./°C' in data.columns:
            features['temperature'] = data['##Temp./°C']
            features['temp_squared'] = data['##Temp./°C'] ** 2
            features['temp_inverse'] = 1 / (data['##Temp./°C'] + 273.15)  # Convert to Kelvin
        
        if 'Time/min' in data.columns:
            features['time'] = data['Time/min']
            features['time_squared'] = data['Time/min'] ** 2
        
        if 'Mass/%' in data.columns:
            features['mass_percent'] = data['Mass/%']
            features['mass_loss'] = 100 - data['Mass/%']
        
        if 'Alpha' in data.columns:
            features['alpha'] = data['Alpha']
            features['alpha_squared'] = data['Alpha'] ** 2
            features['alpha_inverse'] = 1 / (data['Alpha'] + 0.001)  # Avoid division by zero
        
        if 'F_alpha' in data.columns:
            features['f_alpha'] = data['F_alpha']
        
        if 'G_alpha' in data.columns:
            features['g_alpha'] = data['G_alpha']
        
        # Extract experimental conditions from filename or data
        if 'Segment' in data.columns:
            features['segment'] = data['Segment']
        
        # Add interaction features
        if 'temperature' in features.columns and 'alpha' in features.columns:
            features['temp_alpha_interaction'] = features['temperature'] * features['alpha']
        
        if 'temperature' in features.columns and 'mass_percent' in features.columns:
            features['temp_mass_interaction'] = features['temperature'] * features['mass_percent']
        
        return features
    
    def _extract_target(self, data, model_type):
        """Extract target variable based on model type"""
        if model_type == 'regression':
            # For regression, predict activation energy or other continuous values
            if 'Ea' in data.columns:
                return data['Ea']
            elif 'R²' in data.columns:
                return data['R²']
            else:
                # Use alpha as target if no specific target column
                return data['Alpha']
        
        elif model_type == 'classification':
            # For classification, categorize materials or conditions
            if 'Segment' in data.columns:
                return data['Segment']
            else:
                # Create categories based on alpha ranges
                alpha_ranges = pd.cut(data['Alpha'], bins=[0, 0.3, 0.7, 1.0], 
                                    labels=['Low', 'Medium', 'High'])
                return alpha_ranges
        
        else:
            raise ValueError("model_type must be 'regression' or 'classification'")
    
    def _encode_categorical_features(self, features):
        """Encode categorical features for Random Forest"""
        categorical_columns = features.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            features[col] = self.label_encoder.fit_transform(features[col].astype(str))
        
        return features
    
    def train_regression_model(self, X, y, test_size=0.2):
        """Train Random Forest regression model"""
        print("Training Random Forest regression model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        self.regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.regressor.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.regressor.predict(X_train_scaled)
        y_pred_test = self.regressor.predict(X_test_scaled)
        
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
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
            'metrics': {
                'train_r2': train_r2, 'test_r2': test_r2,
                'train_rmse': train_rmse, 'test_rmse': test_rmse
            }
        }
    
    def train_classification_model(self, X, y, test_size=0.2):
        """Train Random Forest classification model"""
        print("Training Random Forest classification model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.classifier.predict(X_train_scaled)
        y_pred_test = self.classifier.predict(X_test_scaled)
        
        # Calculate metrics
        train_accuracy = self.classifier.score(X_train_scaled, y_train)
        test_accuracy = self.classifier.score(X_test_scaled, y_test)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
            'metrics': {
                'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy
            }
        }
    
    def hyperparameter_tuning(self, X, y, model_type='regression'):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("Performing hyperparameter tuning...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if model_type == 'regression':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='r2', n_jobs=-1
            )
            
        else:  # classification
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
        
        grid_search.fit(X_scaled, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def feature_importance_analysis(self, feature_names=None):
        """Analyze and visualize feature importance"""
        if self.regressor is None and self.classifier is None:
            raise ValueError("Model must be trained first")
        
        model = self.regressor if self.regressor is not None else self.classifier
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title('Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def cross_validation_analysis(self, X, y, cv_folds=5, model_type='regression'):
        """Perform cross-validation analysis"""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if model_type == 'regression':
            scores = cross_val_score(
                RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state),
                X_scaled, y, cv=cv_folds, scoring='r2'
            )
            print(f"Cross-validation R² scores: {scores}")
            print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        else:
            scores = cross_val_score(
                RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state),
                X_scaled, y, cv=cv_folds, scoring='accuracy'
            )
            print(f"Cross-validation accuracy scores: {scores}")
            print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def predict_new_data(self, new_data):
        """Make predictions on new data"""
        if self.regressor is None and self.classifier is None:
            raise ValueError("Model must be trained first")
        
        # Prepare new data
        new_features = self._extract_features(new_data)
        new_features = self._encode_categorical_features(new_features)
        new_features_scaled = self.scaler.transform(new_features)
        
        # Make predictions
        model = self.regressor if self.regressor is not None else self.classifier
        predictions = model.predict(new_features_scaled)
        
        return predictions
    
    def save_model(self, filepath):
        """Save the trained model"""
        import joblib
        
        model_data = {
            'regressor': self.regressor,
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'features': self.features.columns.tolist() if hasattr(self, 'features') else None
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        import joblib
        
        model_data = joblib.load(filepath)
        self.regressor = model_data['regressor']
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        
        print(f"Model loaded from {filepath}")

# Example usage functions
def analyze_kinetic_data_with_rf(data_path, model_type='regression'):
    """Complete analysis workflow using Random Forest"""
    
    # Initialize Random Forest
    rf_analyzer = KineticRandomForest(n_estimators=100, max_depth=20)
    
    # Load and prepare data
    features, target = rf_analyzer.load_and_prepare_data(data_path, model_type)
    
    # Perform cross-validation
    cv_scores = rf_analyzer.cross_validation_analysis(features, target, model_type=model_type)
    
    # Hyperparameter tuning
    best_model, best_params = rf_analyzer.hyperparameter_tuning(features, target, model_type)
    
    # Train final model
    if model_type == 'regression':
        results = rf_analyzer.train_regression_model(features, target)
    else:
        results = rf_analyzer.train_classification_model(features, target)
    
    # Feature importance analysis
    importance_df = rf_analyzer.feature_importance_analysis(features.columns)
    
    return rf_analyzer, results, importance_df

def predict_activation_energy(features, target):
    """Predict activation energy using Random Forest"""
    
    rf_model = KineticRandomForest(n_estimators=200, max_depth=15)
    
    # Train model
    results = rf_model.train_regression_model(features, target)
    
    # Feature importance
    importance = rf_model.feature_importance_analysis(features.columns)
    
    return rf_model, results, importance

# Main execution
if __name__ == "__main__":
    # Example usage
    print("Random Forest Kinetic Analysis")
    print("=" * 50)
    
    # You can use this with your existing data files
    # Example: data_path = "data/Filterd Data/A2_filtered/HP0_10_A2_filtered.csv"
    
    # Uncomment and modify the path to your data file
    # rf_analyzer, results, importance = analyze_kinetic_data_with_rf(
    #     "path_to_your_data.csv", model_type='regression'
    # )
