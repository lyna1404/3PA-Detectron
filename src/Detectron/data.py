import torch
import numpy as np
import pandas as pd
from pprint import pprint
# Load model and data using torch.hub
model, data = torch.hub.load('rgklab/pretrained_models', 'uci_heart')

'''
# Retrieve data for 'VA Long Beach'
Q_x = data['Hungary'][0]
Q_y = data['Hungary'][1]  # Make sure to use the correct index for Q_y if it is not at index 0

# Generate indices for sampling
N = 20  # Number of samples to select
idx = np.random.RandomState(seed=0).permutation(len(Q_x))[:N]
print(idx)
# Sample data
Q_x_idx = Q_x[idx]
Q_y_idx = Q_y[idx]

# Create a DataFrame for Q_x_idx
feature_names = [f'feature_{i+1}' for i in range(Q_x_idx.shape[1])]
df_features = pd.DataFrame(Q_x_idx, columns=feature_names)

# Add the target label to the DataFrame
df_features['y_true'] = Q_y_idx

# Save to CSV
df_features.to_csv('ood_hungary_sampled.csv', index=False)
'''





