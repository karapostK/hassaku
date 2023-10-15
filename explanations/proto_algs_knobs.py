import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import torch
from torch import nn
from torch.utils import data

import wandb
from algorithms.base_classes import SGDBasedRecommenderAlgorithm


class PrototypePerturb(ABC, nn.Module):
    def __init__(self, n_prototypes: int, n_masks: int = 1):
        """
        Holds the learned masks that are applied on to all prototypes.
        In the base case where n_masks=1 a single mask is applied to all users (possibly items) in the same way.
        It is possible, however, to define multiple masks e.g. to apply different masks to different user groups or different users
        @param n_prototypes: Length of the mask
        @param n_masks: how many masks should be held by the class.
        """

        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_masks = n_masks

        self.name = 'PrototypePerturb'

        logging.info(f'Built {self.name} model \n'
                     f'- n_prototypes: {self.n_prototypes} \n'
                     f'- n_masks: {self.n_masks} \n')

    @abstractmethod
    def forward(self, in_repr: torch.Tensor, mask_idx: torch.Tensor = None):
        pass


class PrototypeMultiply(PrototypePerturb):
    def __init__(self, n_prototypes: int, n_masks: int = 1):
        """
        Holds the learned multiplication mask that is applied on top of all prototypes.
        In the base case where n_masks=1 then the mask can be applied to all users (possibly items) in the same way.
        It is possible, however, to define multiple masks e.g. to apply to different user groups or different users
        @param n_prototypes: Length of the multiplication mask
        @param n_masks: how many multiplication masks should be held by the class.
        """

        super().__init__(n_prototypes, n_masks)

        params = .25 * torch.randn(self.n_masks, self.n_prototypes) + 2
        self.prototype_knobs = nn.Parameter(params, requires_grad=True)

        self.name = 'PrototypeMultiply'

        logging.info(f'Built {self.name} model \n')

    def forward(self, in_repr: torch.Tensor, mask_idx: torch.Tensor = None):
        """
        Applies the mask to the representations.
        @param in_repr: Shape is [batch_size, n_prototypes]
        @param mask_idx: Shape is [batch_size]
        """
        if mask_idx is None and self.n_masks > 1:
            raise ValueError('Mask indexes are null but there exist multiple masks. Which one to use?')
        elif mask_idx is None and self.n_masks == 1:
            mask_idx = torch.zeros(self.n_prototypes, device=in_repr.device)
        knobs = self.prototype_knobs[mask_idx]  # [batch_size, n_prototypes]
        knobs = nn.Sigmoid()(knobs)  # Restricting the values in [0,1]
        return in_repr * knobs


class PrototypeAdd(PrototypePerturb):
    def __init__(self, n_prototypes: int, n_masks: int = 1):

        super().__init__(n_prototypes, n_masks)

        self.prototype_knobs = nn.Parameter(0.1 * torch.randn(self.n_masks, self.n_prototypes), requires_grad=True)

        self.name = 'PrototypeAdd'

        logging.info(f'Built {self.name} model \n')

    def forward(self, in_repr: torch.Tensor, mask_idx: torch.Tensor = None):
        """
        Applies the mask to the representations.
        @param in_repr: Shape is [batch_size, n_prototypes]
        @param mask_idx: Shape is [batch_size]
        """
        if mask_idx is None and self.n_masks > 1:
            raise ValueError('Mask indexes are null but there exist multiple masks. Which one to use?')
        elif mask_idx is None and self.n_masks == 1:
            mask_idx = torch.zeros(self.n_prototypes, device=in_repr.device)
        knobs = self.prototype_knobs[mask_idx]  # [batch_size, n_prototypes]
        return in_repr + knobs


class TunePrototypeRecModel(SGDBasedRecommenderAlgorithm):

    def __init__(self, prototype_model: SGDBasedRecommenderAlgorithm, entity_name: str, type_perturb: str = 'multiply',
                 user_to_mask: torch.Tensor = None):
        """
        @param prototype_model:
        @param entity_name:
        @param user_to_mask: Shape is [n_users]. Each entry is an index if not None
        @return:
        """
        super().__init__()
        assert entity_name in ['user', 'item'], "Entity name should be either user or item!"

        if entity_name == 'user':
            assert prototype_model.name in ['UProtoMF', 'UIProtoMF', 'ACF', 'ECF']
        else:
            assert prototype_model.name in ['IProtoMF', 'UIProtoMF', 'ACF', 'ECF']

        assert type_perturb in ['multiply', 'add']

        self.prototype_model = prototype_model
        self.entity_name = entity_name
        self.type_perturb = type_perturb

        self.user_to_mask = torch.zeros((self.prototype_model.n_users,),
                                        dtype=torch.int64) if user_to_mask is None else user_to_mask

        n_prototypes = None
        if self.prototype_model.name in ['UProtoMF', 'IProtoMF']:
            n_prototypes = self.prototype_model.n_prototypes
        elif self.prototype_model.name == 'UIProtoMF':
            if self.entity_name == 'user':
                n_prototypes = self.prototype_model.uprotomf.n_prototypes
            else:
                n_prototypes = self.prototype_model.iprotomf.n_prototypes
        elif self.prototype_model.name == 'ACF':
            n_prototypes = self.prototype_model.n_anchors
        elif self.prototype_model.name == 'ECF':
            n_prototypes = self.prototype_model.n_clusters

        n_masks = 1 if self.user_to_mask is None else self.user_to_mask.max() + 1

        if self.type_perturb == 'multiply':
            self.prototype_perturb = PrototypeMultiply(n_prototypes, n_masks)
        elif self.type_perturb == 'add':
            self.prototype_perturb = PrototypeAdd(n_prototypes, n_masks)
        else:
            raise ValueError('How did you get here?')

        self.name = 'TunePrototypeRecModel'

        logging.info(f'Built {self.name} model \n'
                     f'- prototype_model: {self.prototype_model.name} \n'
                     f'- entity_name: {self.entity_name} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:

        mask_idxs = self.user_to_mask[u_idxs]  # [batch_size]

        if self.entity_name == 'item':

            i_repr = self.prototype_model.get_item_representations_pre_tune(i_idxs)
            if type(i_repr) == tuple:
                # The adjustable representation should be in the first position
                i_middle_repr = i_repr[0]
            else:
                i_middle_repr = i_repr

            i_repr_new = self.prototype_perturb(i_middle_repr, mask_idxs)
            if type(i_repr) == tuple:
                i_repr_new = (i_repr_new, *i_repr[1:])
                # todo:make sure everyone returns this
            i_repr = self.prototype_model.get_item_representations_post_tune(i_repr_new)
        else:
            i_repr = self.prototype_model.get_item_representations(i_idxs)

        if self.entity_name == 'user':
            u_repr = self.prototype_model.get_user_representations_pre_tune(u_idxs)
            if type(u_repr) == tuple:
                # The adjustable representation should be in the first position
                u_middle_repr = u_repr[0]
            else:
                u_middle_repr = u_repr

            u_repr_new = self.prototype_perturb(u_middle_repr, mask_idxs)
            if type(u_repr) == tuple:
                u_repr_new = (u_repr_new, *u_repr[1:])
                # todo:make sure everyone returns this
            u_repr = self.prototype_model.get_user_representations_post_tune(u_repr_new)
        else:
            u_repr = self.prototype_model.get_user_representations(u_idxs)

        return self.prototype_model.combine_user_item_representations(u_repr, i_repr)

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        mask_idxs = self.user_to_mask[u_idxs]  # [batch_size]
        if self.entity_name == 'user':
            u_repr = self.prototype_model.get_user_representations_pre_tune(u_idxs)
        else:
            u_repr = self.prototype_model.get_user_representations(u_idxs)
        return u_repr, mask_idxs

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.entity_name == 'item':
            i_repr = self.prototype_model.get_item_representations_pre_tune(i_idxs)
        else:
            i_repr = self.prototype_model.get_item_representations(i_idxs)
        return i_repr

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:

        mask_idxs = u_repr[1]
        u_repr = u_repr[0]

        if self.entity_name == 'item':

            if type(i_repr) == tuple:
                # The adjustable representation should be in the first position
                i_middle_repr = i_repr[0]
            else:
                i_middle_repr = i_repr

            i_repr_new = self.prototype_perturb(i_middle_repr, mask_idxs)
            if type(i_repr) == tuple:
                i_repr_new = (i_repr_new, *i_repr[1:])
            i_repr = self.prototype_model.get_item_representations_post_tune(i_repr_new)

        if self.entity_name == 'user':
            if type(u_repr) == tuple:
                # The adjustable representation should be in the first position
                u_middle_repr = u_repr[0]
            else:
                u_middle_repr = u_repr

            u_repr_new = self.prototype_perturb(u_middle_repr, mask_idxs)
            if type(u_repr) == tuple:
                u_repr_new = (u_repr_new, *u_repr[1:])
            u_repr = self.prototype_model.get_user_representations_post_tune(u_repr_new)

        out = self.prototype_model.combine_user_item_representations(u_repr, i_repr)
        return out

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        pass

    def post_val(self, curr_epoch: int):
        return {'weights': wandb.Histogram(nn.Sigmoid()(self.prototype_perturb.prototype_knobs.cpu().detach()))}
