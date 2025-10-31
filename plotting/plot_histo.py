import os
from read_json_evaluation_result import aggregate
import matplotlib.pyplot as plt
import numpy as np

from statistics import mean, stdev, median, variance

plt.style.use(['science', 'no-latex'])

c_path = '../Toy/Scenario'

xx = []
y1 = []
y1_err = []
y2 = []
y2_err = []

h_index = 5


paths = os.listdir(c_path)

paths = ['CaseA', 'CaseB', 'CaseC', 'CaseD', 'CaseD_Bis', 'CaseG_11_1', 'CaseI_11_1',]


for p in paths:
    # if 'G' not in p:
    #     continue
    rr = aggregate(os.path.join(c_path, p))

    print(p, rr)

    def stampa(k):
        vector = rr['D'][k]
        print(k, mean(vector), median(vector), max(vector), min(vector), stdev(vector), variance(vector))

    # stampa(5)
    # stampa(3)
    # stampa(2)
    stampa(1)

    exit()
    if rr:
        xx.append(p)
        y1.append(mean(rr['H'][h_index]))
        y1_err.append(stdev(rr['H'][h_index]))
        y2.append(mean(rr['D']['avg']))
        y2_err.append(stdev(rr['D']['avg']))


# fig = plt.figure()
# ax1 = plt.subplot(211)
# ax2 = plt.subplot(212, sharex=ax1)

# ax1.bar(xx, y1, yerr=y1_err)
# ax2.bar(xx, y2, yerr=y2_err)

x__ = np.arange(len(xx))

# print(xx)

plt.bar(x__-0.2, y1, yerr=y1_err, width=0.4, label='H@5')
plt.bar(x__+0.2, y2, yerr=y2_err, width=0.4, label='Diff')

plt.xticks(x__, xx)

plt.legend()
plt.savefig('/Users/edoardo/Desktop/Histo.pdf', bbox_inches='tight')

plt.show()
