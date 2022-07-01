import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from matplotlib.cm import get_cmap

cmaps_global = {'grey':"Greys", 'green':"Greens", 'purple':"Purples",'magenta':"RdPu",'blue':"Blues",'red':"Reds",'orange':"YlOrBr",'wt':"Greys", 'residuals':"Greys"}
pixel2um = 0.325

def format_number(x, dec=1):
    x = float(x)
    if x % 1 == 0:
        return int(x)
    else:
        return round(x, dec)

def plot_density_stacked(field, lm, site, names = ['grey','green', 'purple', 'blue', 'red', 'orange', 'wt'],
                         save=False, ax=None, flipped=False, rescale_x=1, grid_mm2=1):
    
    if flipped:
        site = -site

    data = (field[:,:,-site,:] * lm[:,:,-site, None] / grid_mm2 * 1e-3).mean(axis=0)
    data = np.concatenate([data[:,:-3], data[:,[-3,-2]].sum(axis=1)[:,None]], axis=1)
    color_list = [get_cmap(cmaps_global[name])(150) for name in names]
    line_id_list = np.arange(len(names))
    xold = np.linspace(0, data.shape[0],data.shape[0])
    xnew = np.linspace(0, data.shape[0], 5000)

    lines_smooth = []
    for i, idx in enumerate(line_id_list): 
        line = data[:,idx].T

        spl_line = make_interp_spline(xold, line, k=2)
        line_smooth = spl_line(xnew)
        line_smooth[line_smooth < 0] = 0
        lines_smooth.append(line_smooth)

    cum = np.zeros(lines_smooth[0].shape)
    if ax is not None:
        for i in range(len(line_id_list)-1):
            ax.fill_between(xnew / data.shape[0]*rescale_x, cum, cum + lines_smooth[i], color=color_list[i], alpha=1)
            cum += lines_smooth[i]
        ax.plot(xnew / data.shape[0]*rescale_x, cum + lines_smooth[i+1], color='black', lw=1.5, alpha=1)


        #ax.set_xlim(-1)
        ax.set_ylim(0)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
    else:
        for i in range(len(line_id_list)-1):
            plt.fill_between(xnew / data.shape[0] *rescale_x, cum, cum + lines_smooth[i], color=color_list[i], alpha=1)
            cum += lines_smooth[i]
        plt.plot(xnew / data.shape[0]*rescale_x, cum + lines_smooth[i+1], color='black', lw=1.5, alpha=1)


        #plt.xlim(-1)
        plt.ylim(0)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

def plot_field(mut_sample, field, lm, th=0.75,
               names = ['blue','green', 'orange','purple'], ax=None,
               image=None, grid_mm2=None, n_factors=None, n_wt = 2, scale=15):
    if image is None:
        image = mut_sample._scaffold_image
        
    if grid_mm2 is None:
        grid_mm2 = (mut_sample.get_img_size(mut_sample.image)[0] \
                * pixel2um / field.shape[1])**2 / 1e6

    if n_factors is None: 
        n_factors = field.shape[-1]
    
    f = field.mean(0)
    l = lm.mean(0)
    
    fmap = (f[:,:,:n_factors-2]).argmax(2)
    fn = (cv2.blur(l,(3,3)) / grid_mm2 < 300)
    if type(th) is not list:
        fn |= (f[:,:,n_factors-2:]).sum(2) > 0.75
    elif type(th) is list:
        for i, t in enumerate(th):
            fn[(f[:,:,:n_factors-2]).argmax(2) == i] |= ((f[:,:,n_factors-2:]).sum(2) > t)[(f[:,:,:n_factors-2]).argmax(2) == i]
    c = [get_cmap(cmaps_global[n])(150) for n in names] + [(1,1,1,1)] * n_wt
    
    img = image
    img = (img / img.max() * 255).astype(np.uint8)
    s = img.shape
    s = tuple([int(x) for x in list(s)[::-1]])
    p35, p90 = np.percentile(img, (35, 90))
    processed_img = exposure.rescale_intensity(img, in_range=(p35, p90))

    b = cv.resize(processed_img, s)[::-1,:]/255.
    b = np.maximum(np.minimum(b,1),0)
    Fc = np.array([c[int(i)] for i in fmap.flatten()]).reshape((*fmap.shape,-1)).transpose((1,0,2))[::-1,:,:3]
    print(Fc.shape, fn.T.shape)
    Fc[fn.T[::-1,:],:]=1.0
    out = (cv.resize(Fc, s) * b.reshape(*b.shape,1))
    
    if ax is not None:
        ax.imshow(out)
        ax.plot([s[0]*0.95,
             s[0]*0.95 - 2.5e3 / 0.325 / scale ],
             [s[1]*(.95),
              s[1]*(.95)], color='white', lw=5)
        ax.set_axis_off()
    else:
        plt.imshow(out)
        plt.plot([s[0]*0.95,
             s[0]*0.95 - 2.5e3 / 0.325 / scale ],
             [s[1]*(.95),
              s[1]*(.95)], color='white', lw=5)
        plt.axis('off')