# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
r = pd.read_csv("records_.csv")
# %%
r
# %%
plt.plot(r["gamma"], r["m*i"])

# %%
plt.title("Mean value of robustizations")
plt.xlabel("Robustness(gamma)")
plt.ylabel("Mean saved cost")
plt.plot(r["gamma"], r["mvors"])
# %%
plt.title("Improvement of standard deviations")
plt.xlabel("Robustness(gamma)")
plt.ylabel("Improve rate of standard deviation")
plt.plot(r["gamma"], r["iostds"])

# %%
plt.title("Robust rate")
plt.xlabel("Robustness(gamma)")
plt.ylabel("Robust rate")
plt.plot(r["gamma"], r["rr"])
