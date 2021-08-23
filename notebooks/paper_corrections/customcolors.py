from matplotlib.colors import to_rgba

colors = list(map(to_rgba, """
#55115c
#821760
#ab275e
#cd4158
#e7624f
#f88646
#ffad40
#ffd547
""".strip().split('\n'))) #https://learnui.design/tools/data-color-picker.html#palette

colors10 = list(map(to_rgba, """
#55115c
#7d1560
#a1225f
#c1375b
#db5154
#eb694d
#f78347
#fe9d42
#ffb941
#ffd547
""".strip().split('\n'))) #https://learnui.design/tools/data-color-picker.html#palette

#make a colormap from the colors
from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list("mycmap", colors)
