# housing_regression.py
# Linear Regression for House Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_preprocess_data():
    """Load and preprocess the housing data"""
    # Load the dataset
    df = pd.read_csv('Housing.csv')
    
    # Convert categorical variables to numerical (0/1)
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))
    
    # Convert furnishingstatus to numerical (one-hot encoding)
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
    
    return df

def exploratory_data_analysis(df):
    """Perform exploratory data analysis and visualization"""
    # Set style for plots
    sns.set_style('whitegrid')
    
    # 1. Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # 2. Price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True)
    plt.title('Price Distribution')
    plt.savefig('price_distribution.png')
    plt.close()
    
    # 3. Area vs Price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='area', y='price', data=df)
    plt.title('Area vs Price')
    plt.savefig('area_vs_price.png')
    plt.close()

def train_and_evaluate_model(df):
    """Train and evaluate the linear regression model"""
    # Define features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Get coefficients
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Visualization of results
    # 1. Actual vs Predicted prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    # 2. Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig('residuals_plot.png')
    plt.close()
    
    return coefficients, mae, mse, rmse, r2

def main():
    print("Housing Price Prediction using Linear Regression\n")
    
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    print("Data loaded successfully with shape:", df.shape)
    
    # 2. Perform EDA
    print("\nPerforming exploratory data analysis...")
    exploratory_data_analysis(df)
    print("EDA visualizations saved to PNG files")
    
    # 3. Train and evaluate model
    print("\nTraining and evaluating the model...")
    coefficients, mae, mse, rmse, r2 = train_and_evaluate_model(df)
    
    # 4. Display results
    print("\n=== Model Evaluation Results ===")
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"Mean Squared Error (MSE): {mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    print("\n=== Feature Coefficients ===")
    print(coefficients.to_string(index=False))
    
    print("\nVisualizations of results saved to PNG files")
    print("\nTask completed successfully!")

if __name__ == "__main__":
    main()