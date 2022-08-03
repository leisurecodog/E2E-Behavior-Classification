import numpy as np
import matplotlib.pyplot as plt

# x = np.arange(5, 100, 5)
# y1 = [0.0029, 0.007, 0.013, 0.017, 0.023, 0.027, 0.032, 0.037, 0.041, 0.046, 0.048, 0.056, 0.058, 0.063, 0.067, 0.0750, 0.0758, 0.078, 0.081]
# y2 = [0.0000591, 0.0001, 0.0004, 0.0006, 0.001, 0.0014, 0.0019, 0.0026, 0.0031, 0.0039, 0.0047, 0.0057, 0.0065, 0.0075, 0.0086, 0.0098, 0.01, 0.012, 0.013]
# y3 = [0.0015, 0.008, 0.023, 0.046, 0.081, 0.128, 0.202, 0.283, 0.379, 0.502, 0.648, 0.885, 1.081, 1.324, 1.663, 1.976, 2.329, 2.717, 3.164]
# plt.plot(x, y1, "-b", marker='o', label="Compute Adjancency")
# plt.plot(x, y2, "-r", marker='o', label="Compute Laplacian")
# plt.plot(x, y3, '-g', marker='o', label="Compute Eigenvector")
# plt.title("Execution Time for Different Stage of GraphRQI")
# plt.ylabel("Execution Time (s)", fontsize=14)
# plt.xlabel("ID Number", fontsize=14)
# # for idx, i in enumerate(range(5, 100, 5)):
    # # plt.text(i, y1[idx], str(y1[idx]))
    # # plt.text(i, y2[idx], str(y2[idx]))
    # # plt.text(i, y3[idx], str(y3[idx]))
# plt.legend(loc="best", fontsize=14)
# plt.show()
strs =input()

prec, rec = strs.split(' ')
prec, rec = float(prec), float(rec)

print("F1 score:", (2*prec*rec)/(prec+rec))
