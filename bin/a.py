import numpy as np
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

plt.style.use('seaborn')
# classic,fivethirtyeight,ggplot,bmh,dark_background,seaborn
fig = plt.figure()
# ax = Axes3D(fig)
ax = plt.axes()
data = pd.read_csv(r'./out.csv',
                   names=['col1', 'col2', 'col3', 'col4'],
                   sep=',')  # 读取csv数据
sli = 200000
alpha = 0.0003
x = np.linspace(0, sli, sli)
y = np.asarray(data.loc[:, 'col4'][:sli:], float)
# z = np.asarray(data.loc[:, 'col3'][1::], float)
ax.scatter(x, y, s=.5, color=(0., 0., .8))
# ax.plot3D(x, y, z, 'gray')
# ax.scatter3D(x, y, z, 'gray')
# ax.set_xlabel('grad1')
# ax.set_ylabel('grad2')
# ax.set_zlabel('grad3')
# ax.set_title('3D line plot')
plt.title('Cost iter=' + str(sli) + ' alpha=' + str(alpha))
plt.xlabel('iter')
plt.ylabel('J')
# plt.show()
plt.savefig('Figure_4.png', dpi=600)
