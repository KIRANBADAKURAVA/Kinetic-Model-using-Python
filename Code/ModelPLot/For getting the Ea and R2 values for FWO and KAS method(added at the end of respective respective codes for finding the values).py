
# Print the R² and Ea values
for alpha in alpha_values:
    print(f'Alpha = {alpha}: R² = {r2_values[alpha]:.4f}, Ea = {Ea_values[alpha]:.2f} J/mol')
    
        # Prepare data for saving to Excel
output_data = []
for alpha in alpha_values:
    output_data.append({
        'Alpha': alpha,
        'R²': r2_values[alpha],
        'Ea (J/mol)': Ea_values[alpha]
    })
