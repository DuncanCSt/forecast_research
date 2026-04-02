import rpy2.robjects as robjects
import numpy as np
import matplotlib.pyplot as plt

readRDS = robjects.r['readRDS']

def load_rds_array(path):
    """Load an R array into numpy, preserving shape and dimnames."""
    r_obj    = readRDS(path)
    dims     = tuple(int(d) for d in robjects.r['dim'](r_obj))
    arr      = np.array(r_obj).reshape(dims, order='F')
    r_dimnames = robjects.r['dimnames'](r_obj)
    dimnames = []
    for dn in r_dimnames:
        if dn == robjects.rinterface.NULL:
            dimnames.append(None)
        else:
            dimnames.append(list(dn))
    return arr, dimnames

def plot_training_history(history):
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history['loss'], label='Training Loss', marker='o', color='blue')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss', marker='o', color='orange')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history['evidential_kl_divergence'], label='Training KL Divergence', marker='o', color='lightgreen')
    plt.plot(epochs, history.history['val_evidential_kl_divergence'], label='Validation KL Divergence', marker='o', color='pink')
    plt.plot(epochs, history.history['cross_entropy_loss'], label='Training Cross-Entropy Loss', marker='o', color='purple')
    plt.plot(epochs, history.history['val_cross_entropy_loss'], label='Validation Cross-Entropy Loss', marker='o', color='brown')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()