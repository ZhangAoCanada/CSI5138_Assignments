import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from glob import glob

def ReadSamples(file_name):
    """
    Function:
        Read all stored samples
    """
    samples = np.load(file_name)
    return samples


def ReadAllSamples(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size):
    """
    Function:
        Read all files under the current directory
    """
    dir_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + "_" + \
                str(latent_size) + "_" + str(hidden_layer_size) + "/"
    all_files = glob(dir_n + "*.npy")
    num_files = len(all_files)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for file_id in range(1450, num_files):
        file_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + \
                    "_" + str(latent_size) + "_" + str(hidden_layer_size) + "/" + str(file_id) + ".npy"
        current_samples = ReadSamples(file_n)
        if dataset_name == "MNIST":
            current_samples = current_samples.reshape((20, 30, 28, 28))
        else:
            current_samples = current_samples.reshape((20, 30, 32, 32, 3))

        all_imgs = []
        for i in range(5):
            row_imgs = []
            for j in range(10):
                row_imgs.append(current_samples[i, j])
            row_imgs = np.concatenate(row_imgs, axis = 1)
            all_imgs.append(row_imgs)
        all_imgs = np.concatenate(all_imgs, axis = 0)    

        plt.cla()
        ax1.clear()
        ax1.imshow(all_imgs, cmap="gray")
        ax1.axis("off")
        ax1.set_title(str(file_id))
        fig.canvas.draw()
        plt.pause(0.5)



if __name__ == "__main__":
    """
    model names:
        "VAE"
        "GAN"
        "WGAN"
    dataset names:
        "MNIST"
        "CIFAR"
    """
    model_name = "GAN"
    dataset_name = "CIFAR"

    num_hidden = 0
    latent_size = 50
    hidden_layer_size = 256

    ReadAllSamples(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size)


    # model_names = ["VAE", "GAN", "WGAN"]
    # dataset_names = ["MNIST"]

    # num_hiddens = [4, 5]
    # latent_sizes = [10, 50, 100, 200]
    # hidden_layer_sizes = [128, 256, 512, 1024]
    # hidden_layer_size = 256

    # for dataset_name in dataset_names:
    #     for num_hidden in num_hiddens:
    #         for latent_size in latent_sizes:
    #             for model_name in model_names:
    #                 ReadAllSamples(model_name, dataset_name, num_hidden, \
    #                                             latent_size, hidden_layer_size)
        
