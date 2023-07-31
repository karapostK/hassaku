import warnings
from io import BytesIO
from typing import Union

import matplotlib
import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

warnings.simplefilter(action='ignore', category=FutureWarning)

MAX_ENTITIES = 10000


def tsne_plot(dis_mtx: np.ndarray, n_prototypes: int, entity_legend_text: str = 'Entity',
              path_save_fig: Union[str, BytesIO] = None,
              save_fig_format: str = 'png'):
    """
    Creates a TSNE plot to visualize the entity embeddings and the prototypes on a 2d plane.
    @param dis_mtx: Pre-computed distance matrix of prototypes and entities (users or items).
    NB. The prototypes ARE in the first row of the matrix!
    @param n_prototypes: Number of prototypes. The first n_prototypes rows/columns of the distance matrix are considered
    belonging to the prototypes.
    @param entity_legend_text: Text to show in the legend for the Entity
    @param path_save_fig: It can be:
        - None, does not save the figure and calls plt.show()
        - Path, saves the figure
    @param save_fig_format: Used only if path_save_fig is set. Format of the figure to save


    """
    tsne = TSNE(learning_rate='auto', metric='precomputed')

    tsne_results = tsne.fit_transform(dis_mtx)

    tnse_proto = tsne_results[:n_prototypes]
    tsne_entity = tsne_results[n_prototypes:]

    plt.figure(figsize=(6, 6), dpi=200)

    plt.scatter(tsne_entity[:, 0], tsne_entity[:, 1], s=10, alpha=0.6, c='#74add1', label=entity_legend_text)
    plt.scatter(tnse_proto[:, 0], tnse_proto[:, 1], s=30, c='#d73027', alpha=0.9, label='Prototypes')

    plt.axis('off')
    plt.tight_layout()
    plt.legend(loc="upper left", prop={'size': 13})

    if path_save_fig:
        plt.savefig(path_save_fig, format=save_fig_format)
    else:
        plt.show()

    plt.close()



def get_top_k_items(item_weights: np.ndarray, items_info: pd.DataFrame, proto_idx: int,
                    top_k: int = 10, invert: bool = False):
    """
    Used to generate the recommendations to a user prototype or find the closest items to an item prototypes (depending
    on what item_weights encodes).
    :param item_weights: Vector having, for each item, a value for each prototype. Shape is (n_items, n_prototypes)
    :param items_info: a dataframe which contains the item_id field used to look up the item information
    :param proto_idx: index of the prototype
    :param top_k: number of items to return for the prototype, default to 10
    :param invert: whether to look for the farthest items instead of closest, default to false
    :return: a DataFrame containing the top-k closest items to the prototype along with an item weight field.
    """
    assert proto_idx < item_weights.shape[1], \
        f'proto_idx {proto_idx} is too high compared to the number of available prototype'

    weights_proto = item_weights[:, proto_idx]

    top_k_indexes = np.argsort(weights_proto if invert else -weights_proto)[:top_k]
    top_k_weights = weights_proto[top_k_indexes]

    item_infos_top_k = items_info.set_index('item_idx').loc[top_k_indexes]
    item_infos_top_k['item weight'] = top_k_weights
    return item_infos_top_k


def weight_visualization(u_sim_mtx: np.ndarray, u_proj: np.ndarray, i_sim_mtx: np.ndarray, i_proj: np.ndarray,
                         annotate_top_k: int = 3):
    """
    Creates weight visualization plots which is used to explain the recommendation of ProtoMF
    :param annotate_top_k: how many of the highest logits need to be annotated
    """

    rescale = lambda y: 1 - ((y + np.max(y)) / (np.max(y) * 2))  # todo: unsure about this

    def compute_ylims(array):
        y_lim_max = np.max(array) * (1 + 1 / 9)
        y_lim_min = np.min(array) * (1 + 1 / 9)
        return y_lim_min, y_lim_max

    # Computing the logits

    u_prods = u_sim_mtx * i_proj
    i_prods = i_sim_mtx * u_proj

    u_dot = u_prods.sum()
    i_dot = i_prods.sum()

    i_n_prototypes = i_sim_mtx.shape[-1]
    u_n_prototypes = u_sim_mtx.shape[-1]

    # Rescale the plots according to the number of prototypes
    i_vis_ratio = i_n_prototypes / (i_n_prototypes + u_n_prototypes)
    u_vis_ratio = 1 - i_vis_ratio

    # Compute max and mins of the visualization of the logits
    prods_lims = compute_ylims(np.concatenate([u_prods, i_prods]))
    proj_lims = compute_ylims(np.concatenate([u_proj, i_proj]))
    sim_mtx_lims = (0, compute_ylims(np.concatenate([u_sim_mtx, i_sim_mtx]))[1])

    # Plotting the users
    u_fig, u_axes = plt.subplots(3, 1, sharey='row', dpi=100, figsize=(8 * u_vis_ratio, 8))
    u_x = np.arange(u_n_prototypes)

    bars_u_prods = u_axes[0].bar(u_x, u_prods, color=plt.get_cmap('coolwarm')(rescale(u_prods)))
    bars_i_proj = u_axes[1].bar(u_x, i_proj, color=plt.get_cmap('coolwarm')(rescale(i_proj)))
    bars_u_sim_mtx = u_axes[2].bar(u_x, u_sim_mtx, color=plt.get_cmap('coolwarm')(rescale(u_sim_mtx)))

    u_axes[0].set_ylim(prods_lims)
    u_axes[1].set_ylim(proj_lims)
    u_axes[2].set_ylim(sim_mtx_lims)

    u_annotate_protos = np.argsort(-u_prods)[:annotate_top_k]
    for idx, bars in enumerate([bars_u_prods, bars_i_proj, bars_u_sim_mtx]):
        for u_annotate_idx in u_annotate_protos:
            bar = bars[u_annotate_idx]
            label_x = bar.get_x() - 0.8
            label_y = bar.get_height() + (2e-2 if idx == 2 else 1e-2)
            u_axes[idx].annotate(f'{u_annotate_idx}', (label_x, label_y), fontsize=11)

    u_axes[0].set_xlabel(r'$ {\mathbf{s}}^{\mathrm{user}}$', fontsize=24)
    u_axes[1].set_xlabel('$ \hat{\mathbf{t}} $', fontsize=24)
    u_axes[2].set_xlabel('$ \mathbf{u}^{*} $', fontsize=24)
    plt.tight_layout()
    plt.plot()

    # Plotting the items
    i_fig, i_axes = plt.subplots(3, 1, sharey='row', dpi=100, figsize=(i_vis_ratio * 8, 8))
    i_x = np.arange(i_n_prototypes)

    bars_i_prods = i_axes[0].bar(i_x, i_prods, color=plt.get_cmap('coolwarm')(rescale(i_prods)))
    bars_u_proj = i_axes[1].bar(i_x, u_proj, color=plt.get_cmap('coolwarm')(rescale(u_proj)))
    bars_i_sim_mtx = i_axes[2].bar(i_x, i_sim_mtx, color=plt.get_cmap('coolwarm')(rescale(i_sim_mtx)))

    i_axes[0].set_ylim(prods_lims)
    i_axes[1].set_ylim(proj_lims)
    i_axes[2].set_ylim(sim_mtx_lims)

    # Annotations
    i_annotate_protos = np.argsort(-i_prods)[:annotate_top_k]
    for idx, bars in enumerate([bars_i_prods, bars_u_proj, bars_i_sim_mtx]):
        for i_annotate_idx in i_annotate_protos:
            bar = bars[i_annotate_idx]
            label_x = bar.get_x() + (-0.8 if idx == 2 else +0)
            label_y = bar.get_height() + (2e-2 if idx == 2 else 1e-2)
            i_axes[idx].annotate(f'{i_annotate_idx}', (label_x, label_y), fontsize=11)

    i_axes[0].set_xlabel('$ \mathbf{s}^{\mathrm{item}} $', fontsize=24)
    i_axes[1].set_xlabel('$ \hat{\mathbf{u}} $', fontsize=24)
    i_axes[2].set_xlabel('$ \mathbf{t}^{*} $', fontsize=24)
    plt.tight_layout()
    plt.plot()


def protomf_post_val(prototypes, entity_embeddings, sim_func, dis_func, entity_name, curr_epoch):
    """
    Computes:
        - Latent Space projection with TSNE of the entity (user/item) embeddings.
        - Average pair-wise prototype similarity.
        - Entity-to-prototypes statistics (mean, max, min) average across all users.

    """
    n_prototypes = len(prototypes)
    matplotlib.use('agg')
    with torch.no_grad():
        # Sampling entities to avoid GPU crash
        if len(entity_embeddings) >= MAX_ENTITIES:
            indxs = torch.randperm(len(entity_embeddings))[:MAX_ENTITIES]
            entity_embeddings = entity_embeddings[indxs]

        proto_entities = torch.cat([prototypes, entity_embeddings])

        # Computing sim_mtx and dis_mtx
        sim_mtx = sim_func(proto_entities, proto_entities)
        dis_mtx = dis_func(sim_mtx).cpu()

        # Computing average pair-wise prototypes similarity
        sim_mtx_proto = sim_mtx[:n_prototypes, :n_prototypes]

        sim_mtx_proto_tril = torch.tril(sim_mtx_proto, diagonal=-1)
        avg_pairwise_proto_sim = ((sim_mtx_proto_tril.sum() * 2) / (n_prototypes * (n_prototypes - 1))).item()

        # Computing entity-to-prototypes statistics
        entity_to_proto = sim_mtx[n_prototypes:, :n_prototypes]

        entity_to_proto_mean = entity_to_proto.mean(dim=-1).mean().item()
        entity_to_proto_max = entity_to_proto.max(dim=-1).values.mean().item()
        entity_to_proto_min = entity_to_proto.min(dim=-1).values.mean().item()

    # Compute TSNE
    buffer = BytesIO()
    tsne_plot(dis_mtx, n_prototypes, entity_legend_text=entity_name, path_save_fig=buffer, save_fig_format='png')
    buffer.seek(0)
    latent_space_image = wandb.Image(Image.open(buffer), caption="Epoch: {}".format(curr_epoch))

    return {
        'latent_space': latent_space_image,
        'avg_pairwise_proto_sim': avg_pairwise_proto_sim,
        'entity_to_proto_mean': entity_to_proto_mean,
        'entity_to_proto_max': entity_to_proto_max,
        'entity_to_proto_min': entity_to_proto_min,
    }


def protomfs_post_val(prototypes, entity_embeddings, other_entity_embeddings, sim_func, dis_func, entity_name,
                      curr_epoch):
    """
    Computes:
        - Latent Space projection with TSNE of the entity (user/item) embeddings.
        - Average pair-wise prototype similarity.
        - Entity-to-prototypes statistics (mean, max, min) average across all users.
        - Other entity histograms of the weights (the weights multiplied to the similarities)
    """
    post_val_dict = protomf_post_val(prototypes, entity_embeddings, sim_func, dis_func, entity_name, curr_epoch)

    other_entity_embeddings = other_entity_embeddings.detach().cpu().numpy()

    # Computing non-zero weights in the histograms
    bin_weights = other_entity_embeddings.astype(bool).sum(axis=-1)
    bin_weights_mean = bin_weights.mean()
    plt.figure(figsize=(4, 4), dpi=100)
    plt.hist(bin_weights, bins=50)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    bin_weights_image = wandb.Image(Image.open(buffer), caption="Epoch: {}".format(curr_epoch))

    # Computing summed weights in the histograms
    sum_weights = other_entity_embeddings.sum(axis=-1)
    sum_weights_mean = sum_weights.mean()
    plt.figure(figsize=(4, 4), dpi=100)
    plt.hist(sum_weights, bins=50)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    sum_weights_image = wandb.Image(Image.open(buffer), caption="Epoch: {}".format(curr_epoch))

    post_val_dict['bin_weights'] = bin_weights_image
    post_val_dict['sum_weights'] = sum_weights_image
    post_val_dict['bin_weights_mean'] = bin_weights_mean
    post_val_dict['sum_weights_mean'] = sum_weights_mean

    return post_val_dict
