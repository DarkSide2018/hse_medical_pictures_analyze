import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from year_project.telegram_bot.functions import get_skin_problems_dataset

matplotlib.use('TkAgg')

dataframe_train = get_skin_problems_dataset("train")
dataframe_train.drop('target', axis=1, inplace=True)
dataframe_test = get_skin_problems_dataset("test")

scaler = StandardScaler()

scaler.fit(dataframe_train)

data_scaled = scaler.transform(dataframe_train)

dataframe_scaled = pd.DataFrame(data=data_scaled,
                                columns=dataframe_train.columns)

print(dataframe_scaled.head(6))

pca = PCA(n_components=8)

dataframe_pca_transformed = pca.fit_transform(dataframe_scaled)

prop_var = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_

print(f"prop_var : {prop_var}")
print(f"eigenvalues : {eigenvalues}")

PC_numbers = np.arange(pca.n_components_) + 1

plt.plot(PC_numbers,
         prop_var,
         'ro-')
plt.title('Figure 1: Scree Plot', fontsize=8)
plt.ylabel('Proportion of Variance', fontsize=8)
plt.show()