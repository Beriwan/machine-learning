import matplotlib
import matplotlib.pyplot as plt
import KNN

datingDataMat, datingLabels = KNN.file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()