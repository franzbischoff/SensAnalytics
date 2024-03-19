import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

def get_fractal_dimension(data, x_limits, y_limits, debug_plot=False):

    _, _, _, _, x, y = data
    
    #define figure size
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    grid_size = 256
    figsize = (grid_size/(1/px), grid_size/(1/px))
    
    #create plot of CoP
    fig = plt.Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.set_facecolor('#000000')
    lines = ax.plot(x,y,'k',linewidth=1) #plt defines width by points, which is referenced on the figures dpi, that is, 1 point == figure.dpi/72 pixels.
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.axis('off')
    canvas.draw() 
    
    #pixelate plot of CoP in the grid
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_slice = image.reshape(grid_size,grid_size,3)[:,:,0]
    image_slice = 1 - (image_slice - image_slice.min())/(image_slice.max() - image_slice.min()) #0 for background, 1 for plot
    
    #get Minkowski-Bouligand dimension
    f = _fractal_dimension(image_slice)
    
    if debug_plot: 
        ax.axis('on')
        lines[0].set_color("white")
        display(canvas.figure)
        
    return f
    
    
    

def _fractal_dimension(Z, threshold=0.9):
    """Credit: https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1"""
    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
    

