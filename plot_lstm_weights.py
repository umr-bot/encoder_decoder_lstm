# coding: utf-8
import matplotlib.pyplot as plt
weight_index=2
num_weights=len(W[weight_index])//10 # This is just to fit legened on image so
                      # so we only show a certain amount of weights
plt.show()
for c_ind in range(num_weights):
    plt.plot(range(len(W[weight_index][c_ind])), W[weight_index][c_ind], label="W"+str(c_ind))
plt.xlabel("Character indices")
plt.ylabel("weight_amplitude")
plt.legend(loc="upper right")
plt.savefig("figures/M0_W2.pdf")
