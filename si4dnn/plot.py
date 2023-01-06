import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_segmentation(input,prediction,figsize=(10,5),selective_p=None,naive_p=None,title=None):
    fig = plt.figure(figsize = figsize)
    plt.rcParams["font.size"] = 18
    if title is not None:
        fig.suptitle(title)
    if selective_p is not None and naive_p is not None:
        fig.suptitle(f"selective_p:{selective_p:.4f},naive_p:{naive_p:.4f}")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(input[0,:,:,0])
    ax2.imshow(prediction[0,:,:,0])
    ax1.set_title("input")
    ax2.set_title("mask")

def plot_segmentation_with_true(input,prediction,true_output,figsize=(10,5),selective_p=None,naive_p=None,title=None):
    fig = plt.figure(figsize = figsize)
    plt.rcParams["font.size"] = 20
    if title is not None:
        fig.suptitle(title)
    if selective_p is not None and naive_p is not None:
        fig.suptitle(f"selective_p:{selective_p}\nnaive_p:{naive_p}")
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(input[0,:,:,0],cmap=plt.cm.bone)
    ax2.imshow(true_output[0,:,:,0])
    ax3.imshow(prediction[0,:,:,0])
    ax1.set_title("input")
    ax2.set_title("true output")
    ax3.set_title("model output")
    plt.tight_layout()

def plot_p_value_hist(selective_p_values,naive_p_values,file_name,figsize=(10,5)):

    naive_ks = stats.kstest(naive_p_values, 'uniform')
    selective_ks = stats.kstest(selective_p_values, 'uniform')
    naive_fpr = np.count_nonzero(naive_p_values<=0.05)/naive_p_values.shape[0]
    selective_fpr = np.count_nonzero(selective_p_values<=0.05)/selective_p_values.shape[0]
    fig = plt.figure(figsize = figsize)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title(f"SI(ks:{selective_ks.pvalue},FPR:{selective_fpr})")
    ax2.set_title(f"Naive(ks:{naive_ks.pvalue},FPR:{naive_fpr})")
    ax1.hist(selective_p_values,bins=10)
    ax2.hist(naive_p_values,bins=10)
    plt.savefig(file_name)

def plot_real_data_result(input_image,prediction,selective_p_value,naive_p_value,figsize=(10,8)):

    fig = plt.figure()
    plt.rcParams["font.size"] = 20
    fig.suptitle(f"SI P:{selective_p_value:.2f} Naive P:{naive_p_value:.2f}",y=0.95,x=0.528)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plt.rcParams["font.size"] = 15
    ax1.set_title("Image")
    ax2.set_title("Prediction")
    ax1.imshow(input_image[0,:,:,0])
    ax2.imshow(prediction[0,:,:,0]>=0.5)
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_xlabel("")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    # fig.set_title(f"SI P{selective_p_value}\nNaive P:{naive_p_value})")

def plot_real_data(input_image,mask,figsize=(10,8)):

    fig = plt.figure()
    plt.rcParams["font.size"] = 20
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plt.rcParams["font.size"] = 15
    ax1.set_title("Image")
    ax2.set_title("Mask")
    ax1.imshow(input_image[0,:,:,0])
    ax2.imshow(mask[0,:,:,0]>=0.5)
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_xlabel("")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    # fig.set_title(f"SI P{selective_p_value}\nNaive P:{naive_p_value})")
