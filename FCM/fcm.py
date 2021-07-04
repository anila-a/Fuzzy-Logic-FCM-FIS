from fcmeans import FCM
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter
import FCM.dataset as data

x1 = data.mh_x1
x2 = data.mh_x2
x3 = data.mh_x3
x4 = data.mh_x4
x5 = data.mh_x5
x6 = data.mh_x6
x7 = data.mh_x7

# Fit FCM in 3 clusters
fcm = FCM(n_clusters = 3)
fcm.fit(x1)

# Centers and labels
centers = fcm.centers
labels  = fcm.u.argmax(axis = 1)

# Plot result
f, axes = plt.subplots(1, 2)
scatter(x1[:,0], x1[:,1], ax = axes[0])
scatter(x1[:,0], x1[:,1], ax = axes[1], hue = labels)

# Plot centers
scatter(centers[:,0], centers[:,1], ax = axes[1], marker = "s", color = 'r', s = 35)
f.suptitle('Relationship between Age and Organizational Culture', size = 20)
for ax in axes.flat:
    ax.set(xlabel = 'Environment Satisfaction', ylabel = 'Age')

plt.show()
