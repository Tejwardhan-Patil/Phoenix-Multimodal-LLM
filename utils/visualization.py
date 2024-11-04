import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_attention_map(attention_weights, input_sequence, output_sequence, title="Attention Map"):
    """
    Plots an attention map between input and output sequences.
    
    :param attention_weights: 2D numpy array of attention weights.
    :param input_sequence: List of tokens (input sequence).
    :param output_sequence: List of tokens (output sequence).
    :param title: Title for the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=input_sequence, yticklabels=output_sequence, cmap="viridis")
    plt.xlabel('Input Sequence')
    plt.ylabel('Output Sequence')
    plt.title(title)
    plt.show()

def plot_embeddings(embeddings, labels=None, title="Embedding Space", dimension=2):
    """
    Visualizes embeddings in 2D or 3D space.
    
    :param embeddings: 2D numpy array where each row is an embedding.
    :param labels: Optional labels corresponding to each embedding.
    :param title: Title for the plot.
    :param dimension: Dimension of the plot (2 or 3).
    """
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    pca = PCA(n_components=dimension)
    reduced_embeddings = pca.fit_transform(embeddings)

    if dimension == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='b', marker='o')
        if labels is not None:
            for i, label in enumerate(labels):
                plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
        plt.title(title)
        plt.show()
    elif dimension == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c='b', marker='o')
        if labels is not None:
            for i, label in enumerate(labels):
                ax.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], reduced_embeddings[i, 2], label)
        ax.set_title(title)
        plt.show()

def visualize_image_grid(images, rows, cols, title="Image Grid", figsize=(10, 10)):
    """
    Displays a grid of images.
    
    :param images: List of images (as numpy arrays).
    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    :param title: Title for the grid.
    :param figsize: Size of the figure.
    """
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def plot_training_curve(train_loss, val_loss, title="Training Curve"):
    """
    Plots the training and validation loss curves.
    
    :param train_loss: List of training loss values over epochs.
    :param val_loss: List of validation loss values over epochs.
    :param title: Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

def visualize_audio_waveform(audio_data, sample_rate, title="Audio Waveform"):
    """
    Visualizes the waveform of an audio signal.
    
    :param audio_data: Numpy array of the audio signal.
    :param sample_rate: Sample rate of the audio signal.
    :param title: Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    time = np.linspace(0., len(audio_data) / sample_rate, len(audio_data))
    plt.plot(time, audio_data, label="Audio Signal")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()