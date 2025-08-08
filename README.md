# Kinetic Model Analysis Project

A comprehensive Python-based project for thermal kinetic analysis using TGA (Thermogravimetric Analysis) data. This project implements various kinetic modeling methods including Flynn-Wall-Ozawa (FWO), Kissinger-Akahira-Sunose (KAS), and Coats-Redfern (CR) methods to determine activation energies and reaction mechanisms.

## 📁 Project Structure

```
Kinetic-Model-using-Python/
├── Code/
│   ├── DataCleaning&Modification/
│   │   ├── CleanUp.py              # Data cleaning and preprocessing
│   │   ├── DataModification.py     # Data modification utilities
│   │   └── Rough.py                # Initial data processing scripts
│   └── ModelPLot/
│       ├── Finding various alpha F(alpha) G(alpha) and rounding off of alpha values
│       ├── For finding the plot by FWO method
│       ├── For Finding the plot by KAS method
│       ├── For plotting graph and getting Ea and R2 value by CR method
│       ├── For plotting Alpha Vs Temperature at different heating rates.py
│       ├── For finding TGA curve for respective carbon percent at different heating rates
│       ├── For getting the Ea and R2 values for FWO and KAS method
│       └── Rounding of alpha values to closest integer and saving it as a csv file
├── data/
│   ├── AlphaData/                  # Processed alpha values for different conditions
│   │   ├── A2/                     # Avrami-Erofeev model (n=2)
│   │   ├── D1/                     # Diffusion model (1D)
│   │   ├── D2/                     # Diffusion model (2D)
│   │   ├── F0/                     # Reaction order model (n=0)
│   │   ├── F1/                     # Reaction order model (n=1)
│   │   └── R2/                     # Reaction order model (n=2)
│   ├── CombinedData/               # Combined datasets for analysis
│   ├── Filterd Data/               # Filtered and cleaned datasets
│   └── Raw Data/                   # Original TGA data files
```

## 🎯 Project Overview

This project is designed for thermal kinetic analysis of materials using TGA data. It processes raw thermogravimetric data and applies various kinetic modeling approaches to determine:

- **Activation Energy (Ea)**: Energy barrier for the thermal decomposition process
- **Reaction Mechanisms**: Identification of the most suitable kinetic model
- **Kinetic Parameters**: Calculation of F(α) and G(α) functions for different models

## 🔬 Kinetic Models Implemented

The project supports multiple kinetic models for different reaction mechanisms:

### Diffusion Models
- **D1**: One-dimensional diffusion
- **D2**: Two-dimensional diffusion
- **D3**: Three-dimensional diffusion (Jander equation)
- **D4**: Three-dimensional diffusion (Ginstling-Brounshtein equation)

### Reaction Order Models
- **F0**: Zero-order reaction
- **F1**: First-order reaction
- **R2**: Second-order reaction
- **R3**: Third-order reaction

### Nucleation Models
- **A2**: Avrami-Erofeev model (n=2)
- **A3**: Avrami-Erofeev model (n=3)

## 📊 Analysis Methods

### 1. Flynn-Wall-Ozawa (FWO) Method
- Model-free method for activation energy calculation
- Uses the relationship between heating rate and temperature
- Provides reliable Ea values independent of reaction mechanism

### 2. Kissinger-Akahira-Sunose (KAS) Method
- Alternative model-free approach
- Based on the peak temperature at different heating rates
- Validates results from FWO method

### 3. Coats-Redfern (CR) Method
- Model-fitting approach
- Requires assumption of reaction mechanism
- Provides both Ea and pre-exponential factor

## 🚀 Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn statsmodels openpyxl
```

### Data Preparation
1. Place your raw TGA data files in the `data/Raw Data/` directory
2. Ensure data format: Temperature (°C) and Mass (%) columns
3. File naming convention: `HP{condition}_{heating_rate}.txt`

### Running Analysis
1. **Data Cleaning**: Run scripts in `DataCleaning&Modification/` to preprocess your data
2. **Alpha Calculation**: Use the alpha calculation script to determine conversion degree
3. **Model Selection**: Choose appropriate kinetic model based on your material
4. **Kinetic Analysis**: Run FWO, KAS, or CR analysis scripts
5. **Results**: Check generated plots and Excel files for Ea and R² values

### Example Workflow
```python
# 1. Clean your data
python "Code/DataCleaning&Modification/CleanUp.py"

# 2. Calculate alpha values and kinetic functions
python "Code/ModelPLot/Finding various alpha F(alpha) G(alpha) and rounding off of alpha values"

# 3. Perform FWO analysis
python "Code/ModelPLot/For finding the plot by FWO method"

# 4. Perform KAS analysis
python "Code/ModelPLot/For Finding the plot by KAS method"

# 5. Perform CR analysis
python "Code/ModelPLot/For plotting graph and getting Ea and R2 value by CR method"
```

## 📈 Output

The analysis generates:
- **Plots**: Linear regression plots for each method
- **Excel Files**: Tabulated results with Ea and R² values
- **CSV Files**: Processed data for further analysis
- **Statistical Data**: Correlation coefficients and confidence intervals

## 🔧 Customization

### Adding New Kinetic Models
To add a new kinetic model, modify the `FAlpha` function in the alpha calculation script:

```python
elif model == 'NEW_MODEL':
    data_frames[condition_name]["F_alpha"] = your_F_alpha_function
    data_frames[condition_name]["G_alpha"] = your_G_alpha_function
```

### Modifying Analysis Parameters
- Adjust heating rates in the data files
- Modify temperature ranges for analysis
- Change alpha values for specific conversion degrees

## 📚 References

- Flynn, J. H., & Wall, L. A. (1966). A quick, direct method for the determination of activation energy from thermogravimetric data.
- Kissinger, H. E. (1957). Reaction kinetics in differential thermal analysis.
- Coats, A. W., & Redfern, J. P. (1964). Kinetic parameters from thermogravimetric data.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with sample data
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This project is designed for research and educational purposes. Always validate results with experimental data and consult relevant literature for your specific material system.
