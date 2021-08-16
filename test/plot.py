import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plotscatter(XY,label):
    sc = plt.scatter(XY[:, 0], XY[:, 1], vmin=0, vmax=1, c=label, cmap=cm.seismic)
    #plt.colorbar(sc)
    plt.savefig("scatter.png", format="png", dpi=300)