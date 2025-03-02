import matplotlib.pyplot as plt

def vis_grid(frames, r=5, c=5):
    take_every = len(frames) // (r*c)
    frames_vis = frames[::take_every]

    fig, ax = plt.subplots(r,c, figsize=(15,15))
    for i in range(r):
        for j in range(c):
            k = r * i + j
            ax[i,j].imshow(frames_vis[k])
            ax[i,j].set_axis_off()

    fig.tight_layout()
    plt.tight_layout(pad=0.00)
    return fig

