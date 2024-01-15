import matplotlib.pyplot as plt

from year_project.telegram_bot.functions import get_skin_problems_dataset

train = get_skin_problems_dataset("train")

corr = train.corr()

plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.title('Correlation Map')
plt.show()