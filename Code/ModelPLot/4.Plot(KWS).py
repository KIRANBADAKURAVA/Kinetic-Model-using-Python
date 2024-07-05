#KWS

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the Excel file from the CombinedData Folder 
file_path = r'C:\Users\kiran\OneDrive\Desktop\Model Repo\Kinetic-Model-using-Python\Data\CombinedData\CombinedData(0%).xlsx'  
data = pd.read_excel(file_path)

# Add a new column for 1/Temp
data['1/Temp'] = 1 /data['##Temp./°C']

# Filter the dataset for different alpha values
alpha_values = data['Alpha'].unique()

# Create a dictionary to store regression results
regression_results = {}
slopes = {}

# Set up the plot
plt.figure(figsize=(12, 8))

# Loop through each alpha value and create a linear regression model
for alpha in alpha_values:
    # Filter the data for the current alpha value
    alpha_data = data[data['Alpha'] == alpha]
    
    # Extract the 1/Temp and lnB columns
    X = alpha_data['1/Temp']
    y = alpha_data['ln(B/T)']
    
    # Add a constant to the predictor variable matrix (for the intercept)
    X = sm.add_constant(X)
    
    # Create the linear regression model
    model = sm.OLS(y, X).fit()
    
    # Store the results
    regression_results[alpha] = model
    
    # Store the slope (coefficient of 1/Temp)
    slopes[alpha] = model.params['1/Temp']
    
    # Plot the data and the regression line
    sns.scatterplot(x=alpha_data['1/Temp'], y=alpha_data['ln(B/T)'], label=f'Alpha = {alpha}')
    sns.lineplot(x=alpha_data['1/Temp'], y=model.predict(X), label=f'Regression line (Alpha = {alpha})')

#R square value
print(regression_results[0.10].summary())


# Print the slope for each alpha value
for alpha, slope in slopes.items():
    print(f'Alpha = {alpha}: Slope = {slope}')

# Add plot details
plt.xlabel('1/Temperature (1/k)')
plt.ylabel('lnB')
plt.title('Linear Regression of lnB vs 1/Temperature for Different Alpha Values')
plt.legend()
plt.show()

