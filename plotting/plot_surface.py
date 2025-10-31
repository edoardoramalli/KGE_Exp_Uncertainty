import glob
from read_json_evaluation_result import aggregate
from statistics import mean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.style.use('no-latex')

# print(plt.style.available)

folders = glob.glob('./Scenario/CaseF_*')

x1_xx = []
x2_xx = []

zz = []

zz_r = []

for folder in folders:
    folder_split = folder.split('_')
    x1 = float(folder_split[1])
    x2 = float(folder_split[2])

    rr = aggregate(path_case=folder)

    # regression_mean = compute_regression(path=glob.glob(os.path.join(folder, 'Dataset', '*.csv'))[0])

    if rr is None:
        continue

    # score = mean(rr['H'][5])
    # print(rr['D']['avg'])
    score = mean(rr['D']['avg'])

    x1_xx.append(x1)
    x2_xx.append(x2)
    zz.append(score)
    # zz_r.append(regression_mean)


# x1_xx = np.array([x1_xx])
# x2_xx = np.array([x2_xx])
# zz = np.array([zz])


def plot_trisurf(name, z_values):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    zlabel = 'Diff'
    # zlabel = 'H@5'

    ax.view_init(elev=30, azim=45)
    # my_cmap = plt.get_cmap('RdYlGn')
    my_cmap = plt.get_cmap('RdYlGn_r')
    trisurf = ax.plot_trisurf(x1_xx, x2_xx, z_values, linewidth=0.2, antialiased=True, cmap=my_cmap,
                              edgecolor='grey', alpha=0.5)
    plt.colorbar(trisurf, ax=ax, shrink=0.5, aspect=5, label=zlabel)

    # ax.scatter(x1_xx, x2_xx, zz)

    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Bonus: To get rid of the grid as well:
    # ax.grid(False)

    ax.set_zlabel(zlabel)
    # ax.set_ylabel('Number of Dependencies')
    ax.set_ylabel('$\sigma$ Deviation')
    ax.set_xlabel('Number of Uncertainty Value')

    # plt.colorbar().ax.set_ylabel('Label', rotation=120)

    # plt.savefig('/Users/edoardo/Desktop/Surface_Dip_H@5.pdf', bbox_inches='tight')
    plt.savefig('/Users/edoardo/Desktop/Surface_Range_Diff.pdf', bbox_inches='tight')
    plt.title(name)
    plt.show()
    # print(x1_xx)
    # print(x2_xx)
    # print(zz)


plot_trisurf(name='KGE', z_values=zz)
# plot_trisurf(name='Reg', z_values=zz_r)
# plot_trisurf(name='Diff', z_values=[zz[i] - zz_r[i] for i in range(len(zz_r))])
