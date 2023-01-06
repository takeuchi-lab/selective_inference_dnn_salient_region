# %%
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.size"]=20
plt.rcParams['figure.subplot.left'] = 0.15

# %% [markdown]
# # One Reference Image

# %% [markdown]
# ## Synthetic FPR
# - $n\in\{64,256,1024,4096\}$
# - $X_i,X_i^{\text{ref}} \sim N(0,1)$
# - 1000 iteration

# %%
def plot_fpr(df,name,title=None,pos="upper left"):

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    x = [1,2,3,4]


    ax.plot(x,df["parametric_fpr"],label="Proposed",marker="o")
    ax.plot(x,df["oc_fpr"],label="OC",marker="o")
    ax.plot(x,df["bonf_fpr"],label="Bonferroni",marker="o")
    ax.plot(x,df["naive_fpr"],label="Naive",marker="o")


    ax.set_ylabel("False Positive Rate(FPR)")
    ax.set_xlabel("$n$")

    xtickslabel = df["n"]
    ax.set_xticks(x)
    ax.set_xticklabels(xtickslabel)


    if title:
        ax.set_title(title)

    plt.legend(loc=pos)
    fig.savefig(name)

# %% [markdown]
# ### absolute test statistic

# %%
df = pd.read_csv("./result_one_ref/fpr_abs.csv")
sorted_df = df.sort_values(by = "n").iloc[:4,:]
plot_fpr(sorted_df,"./plot_one_ref/fpr_abs.pdf",pos="center right")

# %% [markdown]
# ### average test statistic

# %%
df = pd.read_csv("./result_one_ref/fpr_avg.csv")
sorted_df = df.sort_values(by = "n").iloc[:4,:]
plot_fpr(sorted_df,"./plot_one_ref/fpr_avg.pdf")

# %% [markdown]
# ## Synthetic TPR
# - $n=4096$
# - $\Delta = \{2,4,6,8\}$
# - $X_i =\begin{cases}\Delta + \epsilon \\ \epsilon \end{cases}\;\text{where},\;\epsilon \sim N(0,1)$
# - $X_i^{\text{ref}} ~ N(0,1)$
# 

# %%
def plot_tpr(df,name,title=None,pos="upper left"):

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    x = [1,2,3,4]
    # ax.plot(x,df["naive_tpr"],label="Naive")
    ax.plot(x,df["parametric_tpr"],label="Proposed",marker="o")
    ax.plot(x,df["oc_tpr"],label="OC",marker="o")
    # ax.plot(x,df["permutation1_tpr"],label="Permutation1",marker="o")
    # ax.plot(x,df["permutation2_tpr"],label="Permutation2",marker="o")
    ax.plot(x,df["bonf_tpr"],label="Bonferroni",marker="o")

    ax.set_ylabel("True Positive Rate(TPR)")
    ax.set_xlabel("$\Delta$")

    xtickslabel = df["singal"]
    ax.set_xticks(x)
    ax.set_xticklabels(xtickslabel)

    if title:
        ax.set_title(title)

    plt.legend(loc=pos)
    fig.savefig(name)

# %% [markdown]
# ### absolute test statistic

# %%
df = pd.read_csv("./result_one_ref/tpr_abs.csv")
sorted_df = df.sort_values(by="singal").iloc[:4,:]
plot_tpr(sorted_df,"./plot_one_ref/tpr_abs.pdf")

# %% [markdown]
# ### average test statistic

# %%
df = pd.read_csv("./result_one_ref/tpr_avg.csv")
sorted_df = df.sort_values(by = "singal").iloc[:4,:]
plot_tpr(sorted_df,"./plot_one_ref/tpr_avg.pdf",pos="center right")

# %% [markdown]
# # Multiple Reference Images

# %% [markdown]
# ## Synthetic FPR
# - $n\in\{64,256,1024,4096\}$
# - $X_i,X_i^{\text{ref}} \sim N(0,1)$
# - 1000 iteration

# %% [markdown]
# ### absolute test statistic

# %%
df = pd.read_csv("./result_mul_ref/fpr_abs.csv")
sorted_df = df.sort_values(by = "n").iloc[:4,:]
plot_fpr(sorted_df,"./plot_mul_ref/fpr_abs.pdf",pos="center right")

# %% [markdown]
# ### average test statistic

# %%
df = pd.read_csv("./result_mul_ref/fpr_avg.csv")
sorted_df = df.sort_values(by = "n").iloc[:4,:]
plot_fpr(sorted_df,"./plot_mul_ref/fpr_avg.pdf")

# %% [markdown]
# ## Synthetic TPR
# - $n=4096$
# - $\Delta = \{2,4,6,8\}$
# - $X_i =\begin{cases}\Delta + \epsilon \\ \epsilon \end{cases}\;\text{where},\;\epsilon \sim N(0,1)$
# - $X_i^{\text{ref}} ~ N(0,1)$
# 

# %%


# %% [markdown]
# ### absolute test statistic

# %%
df = pd.read_csv("./result_mul_ref/tpr_abs.csv")
sorted_df = df.sort_values(by = "singal").iloc[:4,:]
plot_tpr(sorted_df,"./plot_mul_ref/tpr_abs.pdf")

# %% [markdown]
# ### average test statistic

# %%
df = pd.read_csv("./result_mul_ref/tpr_avg.csv")
sorted_df = df.sort_values(by = "singal").iloc[:4,:]
plot_tpr(sorted_df,"./plot_mul_ref/tpr_avg.pdf",pos="center right")