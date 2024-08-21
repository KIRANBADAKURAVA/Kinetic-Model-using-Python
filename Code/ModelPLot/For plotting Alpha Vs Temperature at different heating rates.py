import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r'C:\Users\Pranav\Desktop\Final H valuessssss\H0 reduced.xlsx'
data = pd.read_excel(file_path)

plt.figure(dpi=300)

# Plotting
sns.lineplot(data=data, x='##Temp./°C', y='Alpha', hue='b', palette='viridis')

# Add title and labels
plt.title('Alpha vs Temperature with Heating Rate')
plt.xlabel('Temperature(k)')
plt.ylabel('Alpha')

# Show the plot
plt.legend(title='Heating Rate')
plt.show()
