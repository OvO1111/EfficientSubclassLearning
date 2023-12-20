from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA

from torchvision.utils import make_grid
from matplotlib.cm import get_cmap
from utils.parser import OmegaParser
# from meta import db as param
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import math
import os

backends = ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg',
            'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo']


def visualize(
    feature_map, label_map, feat_axis, etype, param,
    impath=None,
    sampling_ratio=20000,
    n_components=2,
    legend=None
):

    if torch.is_tensor(feature_map):
        feature_map = feature_map.cpu().detach().numpy()
    if torch.is_tensor(label_map):
        label_map = label_map.cpu().detach().numpy()
    if not isinstance(feature_map, np.ndarray):
        raise NotImplementedError
    if not isinstance(label_map, np.ndarray):
        raise NotImplementedError
    
    # permute axes, feature vector is along the last dim
    if feat_axis != -1 and feat_axis != feature_map.ndim - 1:
        axes = list(i for i in range(feature_map.ndim))
        axes.pop(feat_axis)
        axes.append(feat_axis)
        feature_map = feature_map.transpose(*axes)
    assert feature_map.shape[:-1] == label_map.shape

    # ignore bg feature vecs
    l_feature_vector = feature_map.shape[-1]
    n_label = len(np.unique(label_map))
    feature_map = feature_map.reshape((-1, l_feature_vector), order='C')
    label_map = label_map.flatten(order='C')
    feature_map = feature_map[label_map > 0]
    label_map = label_map[label_map > 0]

    # sampling an equal number of all subclasses
    if sampling_ratio < 1:
        sampling_ratio = sampling_ratio * len(label_map)
    if sampling_ratio > len(label_map):
        sampling_ratio = len(label_map)
    indices = []
    for i_label in range(1, n_label):
        i_label_map = np.nonzero(label_map == i_label)[0]
        if len(i_label_map) < 2000:
            print(f'not enough labeled pts at label {i_label}, current num={len(i_label_map)}, thresholded at 2000')
            return False
        n_sample = min(sampling_ratio // (n_label - 1), len(i_label_map))
        indices.extend(np.random.choice(i_label_map, n_sample, replace=False))

    feature_map = feature_map[indices]
    label_map = label_map[indices]

    # feature->embedding
    if etype.lower() == 'kpca':
        embedding = KernelPCA(
            kernel='rbf',
            gamma=10,
            degree=5,
            n_components=n_components
        ).fit_transform(feature_map)
    elif etype.lower() == 'tsne':
        embedding = TSNE(
            perplexity=50,
            learning_rate='auto',
            init='pca',
            n_components=n_components
        ).fit_transform(feature_map)
    elif etype.lower() == 'pca':
        embedding = PCA(
            n_components=n_components
        ).fit_transform(feature_map)
    else:
        raise NotImplementedError(f'This {etype} visualization method is not supported')

    # plot and save plot (if not interactive backend)
    fig = plt.figure()
    if embedding.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    if legend is not None:
        scatter = ax.scatter(*embedding.transpose(), c=label_map.tolist(), s=.1, label=[legend[x-1] for x in label_map])
    else:
        scatter = ax.scatter(*embedding.transpose(), c=label_map.tolist(), s=.1)
    legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="fine classes")
    ax.add_artist(legend)
    
    if impath is None:
        plt.savefig(f'{etype}.png')
    if matplotlib.get_backend() in backends:
        plt.show()
    plt.savefig(os.path.join(param.path_to_test, f'{impath}'))
    plt.close()
    plt.clf()

    return True


def gen_colors(n):
    cmap = get_cmap('viridis')
    rgb = [cmap(i)[:-1] for i in np.arange(0, n) / n]
    return rgb


def make_image(writer, param: OmegaParser, image_or_mask, imname, iter_num, n_labels=0, normalize=False, n_grid_images=5, imindex=0):
    label_colors = gen_colors(n_labels)
    dim = math.floor(param.n_dim)
    
    if dim == 3:
        image_or_mask = image_or_mask[imindex]  # take first batch
        if image_or_mask.ndim == 3:
            image_or_mask = image_or_mask.unsqueeze(0)
        n, h, w, d = image_or_mask.shape
        step = int(np.ceil(d / n_grid_images))
        image_or_mask = image_or_mask[..., 0: d: step].permute(3, 0, 1, 2)
        
        if normalize:
            # a MRI instance, take the first mode
            if param.n_channels == 3:
                # treat as a rgb image
                write_im = (image_or_mask[:n_grid_images, 0:3] * 255).to(torch.uint8)
                grid_image = make_grid(write_im, n_grid_images)
                writer.add_image(f'image/{imname}', grid_image, iter_num)
            else:
                write_im = image_or_mask[:n_grid_images, 0:1].repeat(1, 3, 1, 1)
                grid_image = make_grid(write_im, n_grid_images, normalize=True)
                writer.add_image(f'image/{imname}', grid_image, iter_num)
            
        else:
            # a label map instance
            write_im = torch.zeros((n_grid_images, 3, h, w), device=image_or_mask.device)
            if n == 1:
                for i_label in range(1, n_labels):
                    for color_channel in range(3):
                        write_im[:, color_channel] += (image_or_mask[0] == i_label) * label_colors[i_label - 1][color_channel] / n_labels
            
            else:  # one-hot label map
                for i_label in range(1, n_labels):
                    for color_channel in range(3):
                        write_im[:, color_channel] += image_or_mask[:, i_label] * label_colors[i_label - 1][color_channel] / n_labels
            
            grid_image = make_grid(write_im, n_grid_images, normalize=False)
            writer.add_image(f'image/{imname}', grid_image, iter_num)
    
    elif dim == 2:
        if image_or_mask.ndim == 3:
            image_or_mask = image_or_mask.unsqueeze(1)
        b, n, h, w = image_or_mask.shape
        n_grid_images = min(b, n_grid_images)
        
        if normalize:
            # a MRI instance, take the first mode
            if param.n_channels == 3:
                write_im = (image_or_mask[:n_grid_images, 0:3] * 255).to(torch.uint8)
                grid_image = make_grid(write_im, n_grid_images)
                writer.add_image(f'image/{imname}', grid_image, iter_num)
            else:
                write_im = image_or_mask[:n_grid_images, 0:1].repeat(1, 3, 1, 1)
                grid_image = make_grid(write_im, n_grid_images, normalize=True)
                writer.add_image(f'image/{imname}', grid_image, iter_num)
            
        else:
            # a label map instance
            write_im = torch.zeros((n_grid_images, 3, h, w), device=image_or_mask.device)
            if n == 1:
                for i_label in range(1, n_labels):
                    for color_channel in range(3):
                        write_im[:, color_channel] += (image_or_mask[:n_grid_images, 0] == i_label) * label_colors[i_label - 1][color_channel]
            
            else:  # one-hot label map
                for i_label in range(1, n_labels):
                    for color_channel in range(3):
                        write_im[:, color_channel] += image_or_mask[:n_grid_images, i_label] * label_colors[i_label - 1][color_channel]
            
            grid_image = make_grid(write_im, n_grid_images, normalize=False)
            writer.add_image(f'image/{imname}', grid_image, iter_num)
        
        
def make_curve(writer, pred_, gt_, curve_name, n_labels, iter_num):
    assert pred_.shape == gt_.shape
    write_dict = np.zeros((n_labels-1, 3))
    
    for i in range(1, n_labels):
        pred = pred_ == i
        gt = gt_ == i
        if pred.sum() == 0 or gt.sum() == 0:
            continue
        
        tp = torch.bitwise_and(pred, gt).sum()
        fp = torch.bitwise_and(pred, torch.bitwise_not(gt)).sum()
        fn = torch.bitwise_and(torch.bitwise_not(pred), gt).sum()
        
        write_dict[i-1, 0] = 2 * tp / (2 * tp + fp + fn)  # dice
        write_dict[i-1, 1] = tp / (tp + fn)  # recall
        write_dict[i-1, 2] = tp / (tp + fp)  # precision
        
    writer.add_scalars(f'train/{curve_name}_dice', {f'label={i}': write_dict[i-1, 0] for i in range(1, n_labels)}, iter_num)
    writer.add_scalars(f'train/{curve_name}_precision', {f'label={i}': write_dict[i-1, 1] for i in range(1, n_labels)}, iter_num)
    writer.add_scalars(f'train/{curve_name}_recall', {f'label={i}': write_dict[i-1, 2] for i in range(1, n_labels)}, iter_num)


if __name__ == '__main__':
    test_shape = (1, 211, 16, 102, 145)
    test_inputs = np.random.random(test_shape)
    test_labels = np.sum(test_inputs, axis=2)
    test_labels = (test_labels > test_labels.max() * 0.85) * 1 + (test_labels > test_labels.max() * 0.9) * 1

    visualize(test_inputs, test_labels, 2, 'tsne', n_components=3)
    