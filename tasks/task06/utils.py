import os

import tensorflow as tf
import matplotlib.pyplot as plt
import zipfile
import shutil

def plot_data(means_returns):
    x_data = list(range(len(means_returns)))
    plt.plot(x_data, means_returns, 'r-')
    plt.draw()
    plt.pause(0.00001)
    plt.clf()


def save_and_plot(avg_returns, network_saver, baseline_network_saver):
    network_saver.update(avg_returns[-1])
    baseline_network_saver.update(avg_returns[-1])
    # plot_data(avg_returns)



class Saver:
    def __init__(self, initial_threshold, model, name):
        self.best_so_far = initial_threshold
        self.name = name
        self.model = model

    def update(self, mean_episode_return):
        if mean_episode_return > self.best_so_far:
            self.best_so_far = mean_episode_return
            self.model.save(self.name, include_optimizer=False)
            print("Best so far: {}".format(self.best_so_far))
            zip(self.name, self.name)
            shutil.rmtree(self.name)


def load_model(name):
    with zipfile.ZipFile(name + ".zip", 'r') as zip_ref:
        zip_ref.extractall("./"+name)
    return tf.keras.models.load_model(name)


def zip(src, dst):
    zf = zipfile.ZipFile("%s.zip" % (dst), "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src) + 1:]
            print('zipping %s as %s' % (os.path.join(dirname, filename),
                                        arcname))
            zf.write(absname, arcname)
    zf.close()
