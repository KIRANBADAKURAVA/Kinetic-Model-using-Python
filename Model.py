import pandas as pd
import numpy as np
import glob

file_pattern = 'C:/Users/kiran/OneDrive/Desktop/Internship/data/HP*.txt'
file_paths = glob.glob(file_pattern)

data_frames = {}

for file_path in file_paths:
    filename = file_path.split('\\')[-1].split('.')[0]
    conditions = filename.split('_')
    condition_1 = conditions[0][2:]
    condition_2 = conditions[1]
    condition = f'{condition_1}_{condition_2}'
    
    try:
        df = pd.read_csv(file_path, sep=';', header=0, encoding='latin1')
        df["Alpha"] = (100 - df["Mass/%"]) / 1.5053
        data_frames[condition] = df
    
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
alpha=data_frames["0_10"]["Alpha"]
def FAlpha(data_frames, model):
    if model == 'D1':
        data_frames["0_10"]["F_alpha"] = 1 / (2 * alpha)
        data_frames["0_10"]["G_alpha"]= alpha*alpha
    
    
    elif model == 'D2':
        data_frames["0_10"]["F_alpha"] = (-np.log(1 - alpha)) ** (-1)
        data_frames["0_10"]["G_alpha"]=(1 - alpha) * np.log(1 - alpha) + alpha
    
    
    elif model == 'D3':
        part1 = ((3 / 2) * (1 - alpha)) ** (2 / 3)
        part2 = 1 - (1 - alpha) ** (1 / 3)
        
        data_frames["0_10"]["F_alpha"] = part1 * part2 
        
        data_frames["0_10"]["G_alpha"]= (1 - (1 - alpha) ** (1/3)) ** 2
    
    
    elif model == 'D4':
        
        result = (((1 - alpha) ** (-1/3) - 1) ** (-1/3)) * (3 / 2)
        
        data_frames["0_10"]["F_alpha"] = result

        term1 = 1 - (2 * alpha / 3)
        term2 = (1 - alpha) ** (2 / 3)
        result = term1 - term2
        
        data_frames["0_10"]["G_alpha"]=result
    
    elif model == 'F0':
        data_frames["0_10"]["F_alpha"] = 1
        
        data_frames["0_10"]["G_alpha"]=alpha
    

    elif model == 'F1':
        data_frames["0_10"]["F_alpha"] = 1 - alpha
        
        data_frames["0_10"]["G_alpha"]= -np.log(1 - alpha)
    
    elif model == 'R2':
        result = 2 * ((1 - data_frames["0_10"]["Alpha"]) ** (1 / 2))
        
        data_frames["0_10"]["F_alpha"] = result

        data_frames["0_10"]["G_alpha"]= 1 - ((1 - alpha) ** (1/2))

    elif model == 'R3':
        result = 3 * ((1 - data_frames["0_10"]["Alpha"]) ** (2 / 3))
        
        data_frames["0_10"]["F_alpha"] = result

        data_frames["0_10"]["G_alpha"]= 1 - ((1 - alpha) ** (1/3))

    elif model == 'A2':
        term1 = 2 * (1 - data_frames["0_10"]["Alpha"])
        term2 = (-np.log(1 - data_frames["0_10"]["Alpha"])) ** (1 / 2)
        result = term1 * term2
        data_frames["0_10"]["F_alpha"] = result

        data_frames["0_10"]["G_alpha"]= (-np.log(1 - alpha)) ** (1/2)
    
    elif model == 'A3':
        term1 = 3 * (1 - data_frames["0_10"]["Alpha"])
        term2 = (-np.log(1 - data_frames["0_10"]["Alpha"])) ** (2 / 3)
        result = term1 * term2
       
        
        data_frames["0_10"]["F_alpha"] = result
    
        data_frames["0_10"]["G_alpha"]= (-np.log(1 - alpha)) ** (1/3)
    else:
        print(f"Unknown model: {model}")

model = input('Enter the model name: ') 
FAlpha(data_frames, model)


print(data_frames["0_10"])
