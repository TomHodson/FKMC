## improving the behaviour of matplotlib subplots

```
if not hasattr(plt, 'old_subplots'): plt.old_subplots = plt.subplots
@wraps(plt.old_subplots)
def my_subplots(*args, **kwargs):
    if not 'figsize' in kwargs and len(args) > 1:
        width = 5
        maxwidth = 15
        rows, cols, *_ = args
        w = min(maxwidth, width * cols)
        kwargs['figsize'] = (w, w / cols / 1.6 * rows)
    
    gridspec_kw = dict()
    if kwargs.get('sharex') == 'col':
        gridspec_kw['hspace'] = 0.1
    if kwargs.get('sharey') == 'row':
        gridspec_kw['wspace'] = 0.1
    if 'gridspec_kw' in kwargs:
        gridspec_kw.update(kwargs['gridspec_kw'])
    kwargs['gridspec_kw'] = gridspec_kw
    
    return plt.old_subplots(*args, **kwargs)
        
plt.subplots = my_subplots
```

## intersect and interpolate two numpy arrays

```
def solve(f,x):
    s = np.sign(f)
    z = np.where(s == 0)[0]
    if z.size > 0:
        return z
    else:
        s = s[0:-1] + s[1:]
        z = np.where(s == 0)[0]
        return z

def interp(f,x,z):
    m = (f[z+1] - f[z]) / (x[z+1] - x[z])
    return x[z] - f[z]/m
    
x = np.linspace(0,1,100)
f = y1(x) - y2(x)
z = find_zero_crossings(f)
x_intersection = interp_x_position(f, x, z)
y_intersection = np.interp(x_intersection, x, y1(x))

```

## Getting the matplot colour cycle to match line colours
```
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
```