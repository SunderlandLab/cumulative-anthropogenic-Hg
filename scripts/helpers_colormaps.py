import numpy as np
import json as json
import matplotlib as mpl

nature_cmaps = {
    "stone":  [[248, 246, 238], [227, 224, 205], [198, 194, 165], [168, 163, 133], [135, 132, 102], [99, 96, 74]], 
    "grey":   [[229, 230, 237], [201, 206, 218], [153, 162, 180], [111, 123, 145], [72, 86, 108],   [35, 50, 71]], 
    "red":    [[248, 207, 205], [237, 164, 166], [222, 104, 102], [202, 63, 63],   [156, 41, 36],   [115, 26, 20]], 
    "blue":   [[201, 229, 248], [156, 202, 236], [87, 154, 207],  [4, 116, 179],   [2, 80, 144],    [26, 47, 91]], 
    "yellow": [[255, 241, 196], [245, 220, 137], [235, 199, 80],  [205, 159, 45],  [156, 121, 44],  [107, 86, 34]], 
    "green":  [[211, 228, 190], [149, 195, 110], [87, 170, 62],   [61, 137, 46],   [30, 102, 42],   [21, 52, 26]], 
    "purple": [[232, 209, 230], [204, 160, 202], [178, 113, 171], [157, 69, 136],  [110, 39, 105],  [64, 24, 71]], 
    "olive":  [[241, 239, 169], [219, 214, 91],  [192, 191, 41],  [144, 154, 50],  [91, 108, 28],   [47, 61, 23]], 
    "teal":   [[197, 227, 234], [140, 204, 206], [66, 180, 181],  [0, 144, 153],   [0, 93, 110],    [0, 50, 69]], 
    "orange": [[253, 219, 180], [251, 184, 117], [244, 143, 62],  [232, 100, 32],  [173, 71, 35],   [121, 46, 25]],
    }

# convert 0 - 255 rgb to 0 - 1
def rgb_to_1(rgb):
    return [c / 255 for c in rgb]

# create matplotlib colormap from rgb
def create_colormap(name, colors):
    import matplotlib as mpl
    colors = [rgb_to_1(c) for c in colors]
    cm = mpl.colors.ListedColormap([tuple(c) for c in colors])
    cm.name = name
    return cm

# display all colormaps
def display_colormaps(colormaps):
    n = len(colormaps)
    fig, axs = plt.subplots(n, 1, figsize=(6, n * 2))
    for i, cm in enumerate(colormaps):
        plt.sca(axs[i])
        # create an n, 1 array to display the colormap
        x = np.linspace(0, 1, len(cm.colors)).reshape(1, -1)
        plt.imshow(x, cmap=cm, aspect=0.7)
        plt.xticks([])
        plt.yticks([])
        plt.title(cm.name)
    # remove vertical space between subplots
    fig.subplots_adjust(hspace=-0.8)
    plt.show()

def get_nature_cmaps():
    stones  = create_colormap('stone',  nature_cmaps['stone'])
    greys   = create_colormap('grey',   nature_cmaps['grey'])
    reds    = create_colormap('red',    nature_cmaps['red'])
    blues   = create_colormap('blue',   nature_cmaps['blue'])
    yellows = create_colormap('yellow', nature_cmaps['yellow'])
    greens  = create_colormap('green',  nature_cmaps['green'])
    purples = create_colormap('purple', nature_cmaps['purple'])
    olives  = create_colormap('olive',  nature_cmaps['olive'])
    teals   = create_colormap('teal',   nature_cmaps['teal'])
    oranges = create_colormap('orange', nature_cmaps['orange'])
    return {'stones': stones, 'greys': greys, 'reds': reds, 'blues': blues, 'yellows': yellows, 'greens': greens, 'purples': purples, 'olives': olives, 'teals': teals, 'oranges': oranges}


def display_all_colormaps():

    cmaps = get_nature_cmaps()

    display_colormaps([
        # main background
        cmaps['stones'],
        cmaps['greys'],
        # main accents
        cmaps['reds'],
        cmaps['blues'],
        cmaps['yellow'],
    ])

    display_colormaps([
        # extended palettes
        cmaps['olives'],
        cmaps['greens'],
        cmaps['teals'],
        cmaps['blues'],
        cmaps['purples'],
        cmaps['reds'],
        cmaps['oranges'],
        cmaps['yellows'],
    ])

