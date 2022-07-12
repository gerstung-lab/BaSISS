import matplotlib.pyplot as plt


def boxplot_annotate_brackets(num1, num2, center, height, data, dh=.05, barh=0, fs=None, maxasterix=4, p_star=0.05,
                              ax=None):
    """Annotate barplot with p-values, adopted from
     https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph

    Parameters
    ----------
    num1 : int
        Number of left bar to put bracket over
    num2 : int
        Number of right bar to put bracket over
    center : list
        Centers of all bars (like plt.bar() input)
    height : list
        Heights of all bars (like plt.bar() input)
    data : str
        String to write or number for generating asterixes
    dh : float
        Height offset over bar in axes coordinates (0 to 1)
    barh : float
        Extra bar height
    fs : float
        Font size
    maxasterix : int
        maximum number of asterixes to write (for very small p-values)
    p_star : float
        One star unit correspondence to p-value
    ax : matplotlib.axes._subplots.AxesSubplot
        Subplot axis to attach the plot to

    Returns
    -------

    """

    """ 
    

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = p_star

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh)

    if ax is not None:
        ax.plot(barx, bary, c='black', lw=1)
    else:
        plt.plot(barx, bary, c='black', lw=1)

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    if ax is not None:
        ax.text(*mid, text, **kwargs)
    else:
        plt.text(*mid, text, **kwargs)
