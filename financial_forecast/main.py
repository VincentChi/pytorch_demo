import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns

training_input = pd.read_csv('dataset/training_input.csv')
#training_input.info()
print(training_input.shape)
print(training_input.head())
training_output = pd.read_csv('dataset/training_output.csv')

# plt.scatter(training_input.x1, training_input.x2)
# plt.xlabel("Education")
# plt.ylabel("Income")
# plt.show()

# plt.pie(training_output.y1, training_output.y2, training_output.y3)
# plt.show()

