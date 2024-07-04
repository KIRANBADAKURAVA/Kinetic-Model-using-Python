import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel file into a pandas DataFrame
df = pd.read_excel(r"C:\Users\kiran\OneDrive\Desktop\Model Repo\cdata.xlsx")

df['1/T']=1/(df['##Temp./°C']+273)
df['B/Ts']=df['lnB']/((df['##Temp./°C']+273)*(df['##Temp./°C']+273))
# Assuming '##Temp./°C', 'lnB', and 'Alpha' are actual column names in your DataFrame
sns.lineplot(x='1/T', y='B/Ts', hue='Alpha', data=df, legend=True)

# Optionally, add a title
plt.title('Line Plot of lnB vs. ##Temp./°C')

# Show the plot
plt.show()