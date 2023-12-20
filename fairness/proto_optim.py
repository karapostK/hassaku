import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import torch
from torch import nn
from torch.utils import data

from algorithms.base_classes import SGDBasedRecommenderAlgorithm


class PrototypePerturb(ABC, nn.Module):
    def __init__(self, n_prototypes: int, n_groups: int = None, optimize_parameters: bool = True):
        """
        Holds the perturbations that are applied to the prototypes. These can be also learned.
        When n_groups == 1 then the perturbations are applied to all users in the same way.
        When n_groups > 1 then the perturbations are applied differently to different user groups.
        @param n_prototypes: Defines the length of the perturbation parameters
        @param n_groups: How many sets of perturbation parameters should be held by the class.
        @param optimize_parameters: Whether the parameters should be optimized or not.
        """

        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_groups = n_groups
        self.optimize_parameters = optimize_parameters

        self.name = 'PrototypePerturb'

        logging.info(f'Built {self.name} model \n'
                     f'- n_prototypes: {self.n_prototypes} \n'
                     f'- n_groups: {self.n_groups} \n'
                     f'- optimize_parameters: {self.optimize_parameters} \n')

    @abstractmethod
    def forward(self, in_repr: torch.Tensor, group_idx: torch.Tensor = None):
        pass


class PrototypeMultiply(PrototypePerturb):
    """
    This class is used to multiply the prototypes with a lambda parameter.
    NB. n_groups == 1 is the baseline (all users belong to a single group)
    """

    def __init__(self, n_prototypes: int, n_groups: int = 1, optimize_parameters: bool = True,
                 default_parameters: torch.Tensor = None, use_sigmoid: bool = False):

        super().__init__(n_prototypes, n_groups, optimize_parameters)

        assert n_groups > 0, "n_groups should be greater than 0!"

        self.use_sigmoid = use_sigmoid

        if default_parameters is None:
            # Lambdas are initialized close to a sigmoid value of 1 (i.e. mean of sig(x) is 0.88)
            self.lambdas = nn.Parameter(0.25 * torch.randn(self.n_groups, self.n_prototypes) + 2)
        else:
            self.lambdas = default_parameters.clone()

        self.lambdas.requires_grad_(self.optimize_parameters)

        self.name = 'PrototypeMultiply'

        logging.info(f'Built {self.name} model \n'
                     f'- use_sigmoid: {self.use_sigmoid} \n'
                     f'- using default_parameters: {default_parameters is not None} \n')

    def forward(self, in_repr: torch.Tensor, group_idx: torch.Tensor = None):
        """
        @param in_repr: Shape is [batch_size, n_prototypes] or [batch_size,1,n_prototypes]
        @param group_idx: Shape is [batch_size]
        """
        if group_idx is None:
            if self.n_groups > 1:
                raise ValueError('Group indexes are null but there exist multiple groups. Which one to use?')
            else:
                group_idx = torch.zeros(self.n_prototypes, device=in_repr.device, dtype=torch.int64)

        # Retrieving the lambdas for the current group
        lambdas = self.lambdas[group_idx]  # [batch_size, n_prototypes]

        if self.use_sigmoid:
            lambdas = nn.Sigmoid()(lambdas)
        # Multiplying the representations with the lambdas
        return in_repr * lambdas


class PrototypeAdd(PrototypePerturb):
    """
    This class is used to add a delta parameter to the prototypes.
    """

    def __init__(self, n_prototypes: int, n_groups: int = 1, optimize_parameters: bool = True,
                 default_parameters: torch.Tensor = None):

        super().__init__(n_prototypes, n_groups, optimize_parameters)

        if default_parameters is None:
            self.deltas = nn.Parameter(.01 * torch.randn(self.n_groups, self.n_prototypes))
        else:
            self.deltas = default_parameters.clone()

        self.deltas.requires_grad_(self.optimize_parameters)
        self.name = 'PrototypeAdd'

        logging.info(f'Built {self.name} model \n')

    def forward(self, in_repr: torch.Tensor, group_idx: torch.Tensor = None):
        """
        @param in_repr: Shape is [*, batch_size, n_prototypes]
        @param group_idx: Shape is [batch_size]
        """
        if group_idx is None:
            if self.n_groups > 1:
                raise ValueError('Group indexes are null but there exist multiple groups. Which one to use?')
            else:
                group_idx = torch.zeros(self.n_prototypes, device=in_repr.device, dtype=torch.int64)

        # Retrieving the deltas for the current group
        deltas = self.deltas[group_idx]  # [batch_size, n_prototypes]

        # Adding the deltas to the representations
        return in_repr + deltas


class PrototypeTuner(SGDBasedRecommenderAlgorithm):
    """
    This class is used to tune the prototypes of prototype-based models.
    Currently. two approaches are supported:
    - Multiply: a value lambda is multiplied to the prototype
    - Add: a value delta is added to the prototype
    The tuning can be carried out on either the user or the item prototypes.
    NB. Not all models support tuning of both user and item prototypes. (e.g. UProtoMF, IProtoMF)
    Furthermore, the tuning can be carried out differently for different users. This is done by defining a mapping of
    user indexes to group indexes. Each group, then PrototypeTuner gets different tuning parameters.
    E.g. applying different tuning parameters for different user groups based on gender.
    """

    def __init__(self, prototype_based_model: SGDBasedRecommenderAlgorithm,
                 entity_name: str, prototype_perturb: PrototypePerturb,
                 user_idx_to_group: torch.Tensor = None):
        """
        @param prototype_based_model: The model whose prototypes will be tuned
        @param entity_name: Either 'user' or 'item'
        @param prototype_perturb: The perturbation that will be applied to the prototypes
        @param user_idx_to_group: Shape is [n_users]. Each entry is an index if not None. If None, then all users are tuned in the same way.
        """
        super().__init__()
        assert entity_name in ['user', 'item'], "Entity name should be either user or item!"

        if entity_name == 'user':
            assert prototype_based_model.name in ['UProtoMF', 'UIProtoMF', 'ACF', 'ECF']
        else:
            assert prototype_based_model.name in ['IProtoMF', 'UIProtoMF', 'ACF', 'ECF']

        self.prototype_based_model = prototype_based_model
        self.entity_name = entity_name
        self.prototype_perturb = prototype_perturb
        self.user_idx_to_group = user_idx_to_group

        if self.user_idx_to_group is None:
            # If no user to mask mapping is provided, then all users are masked in the same way
            self.user_idx_to_group = torch.zeros(self.prototype_based_model.n_users, dtype=torch.int64)

        self.name = 'PrototypeTuner'

        logging.info(f'Built {self.name} model \n'
                     f'- prototype_based_model: {self.prototype_based_model.name} \n'
                     f'- entity_name: {self.entity_name} \n'
                     f'- prototype_perturb: {self.prototype_perturb.name} \n'
                     )

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:

        i_repr = self.get_item_representations(i_idxs)
        u_repr = self.get_user_representations(u_idxs)
        return self.combine_user_item_representations(u_repr, i_repr)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:

        if self.entity_name == 'item':
            i_repr = self.prototype_based_model.get_item_representations_pre_tune(i_idxs)
        else:
            i_repr = self.prototype_based_model.get_item_representations(i_idxs)

        return i_repr

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:

        group_idxs = self.user_idx_to_group[u_idxs]  # [batch_size]

        if self.entity_name == 'user':
            u_repr = self.prototype_based_model.get_user_representations_pre_tune(u_idxs)
        else:
            u_repr = self.prototype_based_model.get_user_representations(u_idxs)
        return u_repr, group_idxs

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:

        group_idxs = u_repr[1]
        u_repr = u_repr[0]

        if self.entity_name == 'item':
            # --- Item Representation --- #
            if type(i_repr) == tuple:
                # The tunable representation should be in the first position
                i_middle_repr = i_repr[0]
            else:
                i_middle_repr = i_repr

            if self.training:
                # [batch_user, batch_items, n_prototypes]
                i_middle_repr.transpose_(0, 1)  # [batch_items, batch_user, n_prototypes]
                i_repr_new = self.prototype_perturb(i_middle_repr,
                                                    group_idxs)  # [batch_user, batch_items, n_prototypes]
                i_repr_new.transpose_(0, 1)  # [batch_user, batch_items, n_prototypes]
            else:
                # [n_items, n_prototypes]
                i_middle_repr.unsqueeze_(1)  # [n_items, 1, n_prototypes]
                i_repr_new = self.prototype_perturb(i_middle_repr,
                                                    group_idxs)  # [batch_user, batch_items, n_prototypes]
                i_repr_new.transpose_(0, 1)  # [batch_user, batch_items, n_prototypes]

            if type(i_repr) == tuple:
                i_repr_new = (i_repr_new, *i_repr[1:])
            i_repr = self.prototype_based_model.get_item_representations_post_tune(i_repr_new)

        else:
            if type(u_repr) == tuple:
                # The tunable representation should be in the first position
                u_middle_repr = u_repr[0]
            else:
                u_middle_repr = u_repr

            u_repr_new = self.prototype_perturb(u_middle_repr, group_idxs)

            if type(u_repr) == tuple:
                u_repr_new = (u_repr_new, *u_repr[1:])
            u_repr = self.prototype_based_model.get_user_representations_post_tune(u_repr_new)

        return self.prototype_based_model.combine_user_item_representations(u_repr, i_repr)

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        raise NotImplementedError("This model cannot be loaded from conf!")
