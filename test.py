import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
np.random.seed(42)
n = 200

# Generate synthetic features
age = np.random.normal(loc=35, scale=10, size=n)
income = np.random.normal(loc=50000, scale=15000, size=n)
purchase_amount = np.random.exponential(scale=300, size=n)
purchase_count = np.random.poisson(lam=3, size=n)

# Inject a few outliers
income[5] = 250000
purchase_amount[10] = 5000

perTile25=np.percentile(purchase_amount,25)
perTile75=np.percentile(purchase_amount,75)
print(np.where(purchase_amount<perTile25)[0],np.where(perTile75<purchase_amount)[0])

#Part 1
fig=plt.figure(figsize=(24,4))
gs=gridspec.GridSpec(1,3,figure=fig)
p1=fig.add_subplot(gs[0,0])
p2=fig.add_subplot(gs[0,1])
p3=fig.add_subplot(gs[0,2])
p1.hist(age)
p2.hist(income)
p3.hist(purchase_amount)
plt.show()

# Part 2
# fig=plt.figure(figsize=(16,4))
# gs=gridspec.GridSpec(1,2,figure=fig)
# axs1=fig.add_subplot(gs[0,0])
# axs2=fig.add_subplot(gs[0,1])
# #axs3=fig.add_subplot(gs[0,2])
# axs1.boxplot(income)
# axs1.set_title("Income")
# axs2.boxplot(purchase_amount)
# axs2.set_title("Purchase Amount")
# #axs3.boxplot(np.sqrt(purchase_amount))
# #axs3.set_title("Purchase Amount with log scale")
# print("25 percentile: ",perTile25," ",math.log2(perTile25)," ",math.sqrt(perTile25))
# print("75 percentile: ",perTile75," ",math.log2(perTile75)," ",math.sqrt(perTile75))
# plt.show()

# Part3
# scale=plt.figure(figsize=(16,4))
# gs=gridspec.GridSpec(1,2,figure=scale)
# p1=scale.add_subplot(gs[0,0])
# p2=scale.add_subplot(gs[0,1])

# p1.scatter(age,purchase_amount)
# p1.set_xlabel("age")
# p1.set_ylabel("purchase_amount")
# p1.set_title("age vs purchas_amount")
# p2.scatter(income,purchase_amount)
# p2.set_xlabel("income")
# p2.set_ylabel("purchase_amount")
# p2.set_title("income vs purchas_amount")
# plt.show()