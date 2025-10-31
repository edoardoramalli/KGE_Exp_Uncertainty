import json
import matplotlib.pyplot as plt
import glob

# plt.style.use(['no-latex'])
# plt.style.use(['science','ieee', 'no-latex'])
plt.style.use(['science', 'no-latex'])

# paths = glob.glob('./Scenario/CaseA/*/*/results.json')
fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()

c = ['r', 'b', 'y', 'g', 'm', 'c']

# print(paths)

paths = [


    './Scenario/CaseA/16/results.0_0/results.json',
    './Scenario/CaseA/32/results.1_0/results.json',
    './Scenario/CaseA/64/results.2_0/results.json',
    './Scenario/CaseA/128/results.3_0/results.json',
    './Scenario/CaseA/256/results.4_0/results.json',
    './Scenario/CaseA/512/results.5_0/results.json',
 ]

label = [16, 32, 64, 128, 256, 512]

ax1.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
# ax1.yaxis.major.formatter._useMathText = True

ax1.set_xlabel('# Epoch')
ax1.set_ylabel('Validation Loss')
# ax2.set_ylabel('Validation Loss')

for index, path in enumerate(paths):

    with open(path, 'r') as f:
        r = json.load(f)


    y_losses = r['losses']
    x_losses = list(range(0, len(y_losses)))

    y_validation = r['stopper']['results']
    f_validation = r['stopper']['frequency']
    x_validation = list(range(f_validation, len(y_validation) * f_validation + f_validation, f_validation))

    # print(y_losses)
    #
    # exit()


    # ax1.plot(x_losses, y_losses, '{}'.format(c[index % 6]), label=label[index])
    ax1.plot(x_validation, y_validation, '{}--'.format(c[index % 6]), label=label[index])


plt.legend()
plt.savefig("/Users/edoardo/Desktop/Validation.pdf", bbox_inches='tight')
plt.show()