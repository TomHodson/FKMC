from matplotlib import pyplot as plt

columnwidth = 246/72.27

plt.rc('font', family='sans-serif')
plt.rc('figure', dpi=200) #makes the plots display larger in the jupyter view
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=8, direction='in', bottom=True, top=True)
plt.rc('ytick', labelsize=8, direction='in', left=True, right=True)
plt.rc('axes', labelsize=8)
plt.rc('figure', autolayout=False)

from customcolors import colors, colors10, custom_cmap