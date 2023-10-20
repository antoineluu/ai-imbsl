import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
with open("./notebooks/output_drug1.txt", mode="r") as f:
    lines = f.readlines()
    values =np.empty((len(lines),2))
    for i,line in enumerate(lines):
        line = line.replace("\n","").split(sep="   ")
        values[i] = line
    plt.scatter(values[:,0],values[:,1])
    plt.xlabel("Input drug feature 1", size=20)
    plt.ylabel("Autoencoded", size=20)
    
plt.plot(np.array([np.min(values[:,0]),np.max(values[:,0])]),np.array([np.min(values[:,0]),np.max(values[:,0])]), "r")
plt.title("reconstruction MSE: 0.1548", fontsize=20)
plt.savefig("./figures/ae_drug.png")
plt.show()