import matplotlib.pyplot as plt
import pickle


def plot_curves(lss):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.plot(lss['test']['acc_bw'], 'b', label='Greyscale')
    ax.plot(lss['test']['acc_color'], 'r', label='Colored')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    plt.pause(0.001)


def save_dict(fname, data):
    print(f'Save dict to {fname}')
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def load_dict(fname):
    print(f'Load dict from {fname}')
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)
    return data