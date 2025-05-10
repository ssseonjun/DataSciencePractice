import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as girdspec

# Make bmi array for 100 students #
wt=np.random.random(100)*60+40
ht=np.random.random(100)*60+140
bmi=wt/((ht*0.01)**2)
students=np.arange(100)
level=("Under\nweight","Healthy","Over\nweight","Obese")
distribution=(np.sum(bmi<18.5),
   np.where((18.5<=bmi)&(bmi<25.0))[0].size,
   np.count_nonzero((25.0<=bmi)&(bmi<30.0)),
   np.sum(30.0<=bmi))

# show a scatter plot #
fig=plt.figure(figsize=(16,8))
gs=girdspec.GridSpec(2,3,figure=fig)
axs1=fig.add_subplot(gs[0,0])
axs2=fig.add_subplot(gs[0,1])
axs3=fig.add_subplot(gs[0,2])
axs4=fig.add_subplot(gs[1,0])
axs5=fig.add_subplot(gs[1,1])
plt.subplots_adjust(hspace=0.7)

# bar chart for bmi#
axs1.bar(level, distribution)
axs1.set_title("Bar Chart")
axs1.set_xlabel("Level")
axs1.set_ylabel("students")
# histogram for bmi #

cnt,edges,_ = axs2.hist(bmi,bins=(np.min(bmi), 18.5, 25.0, 30.0, np.max(bmi)), edgecolor='black')
edge_center = .5*(edges[1:]+edges[:-1])
axs2.set_title("Histogram")
axs2.set_xticks(edge_center,level)
axs2.set_xlabel("bmi")
axs2.set_ylabel("students")

# pie chart for bmi #
axs3.pie(distribution,labels=level,autopct='%1.2f%%')
axs3.set_title("Pie Chart")


axs4.scatter(bmi, ht, color='r')
axs4.set_title("height and bmi")
axs4.set_xlabel("bmi")
axs4.set_ylabel("height")
 # scatter plot for weight #
axs5.scatter(bmi,wt, color='b')
axs5.set_title("weight and bmi")
axs5.set_xlabel("bmi")
axs5.set_ylabel("weight")

plt.show()

# fig, sct=plt.subplots(1,2,figsize=(10.5,4))
# # scatter plot for height #
# sct[0].scatter(bmi, ht, color='r')
# sct[0].set_title("height and bmi")
# sct[0].set_xlabel("bmi")
# sct[0].set_ylabel("height")
#  # scatter plot for weight #
# sct[1].scatter(bmi,wt, color='b')
# sct[1].set_title("weight and bmi")
# sct[1].set_xlabel("bmi")
# sct[1].set_ylabel("weight")

# plt.show()
