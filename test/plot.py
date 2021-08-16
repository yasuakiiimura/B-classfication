import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plotscatter(XY,label):
    sc = plt.scatter(XY[:, 0], XY[:, 1], vmin=0, vmax=1, c=label, cmap=cm.seismic)
    #plt.colorbar(sc)
    plt.savefig("scatter.png", format="png", dpi=300)
    
def result_draw(test_x,test_t,rs):
    plt.cla()
    r_00_x = [x for (x,r,t) in zip(test_x[:,0],rs[:,0],test_t) if (r == t)and(r == 0)]
    r_00_y = [y for (y,r,t) in zip(test_x[:,1],rs[:,0],test_t) if (r == t)and(r == 0)]
    plt.scatter(r_00_x,r_00_y,c='red',marker='o')
    r_01_x = [x for (x,r,t) in zip(test_x[:,0],rs[:,0],test_t) if (r != t)and(r == 0)]
    r_01_y = [y for (y,r,t) in zip(test_x[:,1],rs[:,0],test_t) if (r != t)and(r == 0)]
    plt.scatter(r_01_x,r_01_y,c='red',marker='x')
    r_11_x = [x for (x,r,t) in zip(test_x[:,0],rs[:,0],test_t) if (r == t)and(r == 1)]
    r_11_y = [y for (y,r,t) in zip(test_x[:,1],rs[:,0],test_t) if (r == t)and(r == 1)]
    plt.scatter(r_11_x,r_11_y,c='blue',marker='o')
    r_10_x = [x for (x,r,t) in zip(test_x[:,0],rs[:,0],test_t) if (r != t)and(r == 1)]
    r_10_y = [y for (y,r,t) in zip(test_x[:,1],rs[:,0],test_t) if (r != t)and(r == 1)]
    plt.scatter(r_10_x,r_10_y,c='blue',marker='x')
    plt.savefig("result.png", format="png", dpi=300)
    return (len(r_00_x)+len(r_11_x))/len(test_x)
