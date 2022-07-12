from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd


class Histogenomic_associations:
    '''
    We store histological annotations, phylogenetic composition and extra information
    '''

    def __init__(self,
                 histo_df: pd.DataFrame,
                 composition_dict: dict,
                 extra_dfs: Optional[Dict[str, pd.DataFrame]] = None
                 ):

        self.histology_df = histo_df  # .astype("category")
        self.histology_df.index = histo_df.index.astype(str)

        self.composition_dict = composition_dict

        # add extra tables and unify their indices
        self.extra_dfs = extra_dfs
        if extra_dfs is None:
            self.extra_dfs_names = None
        else:
            self.extra_dfs_names = list(self.extra_dfs.keys())
            for extra_dfs_name in self.extra_dfs_names:
                self.extra_dfs[extra_dfs_name].index = self.extra_dfs[extra_dfs_name].index.astype(str)

        self.n_subclones = self.composition_dict[list(self.composition_dict.keys())[0]].shape[1]

        self.subset_ids()

    def __getitem__(self, item):
        if item in self.histology_df.index:
            hist = self.histology_df.loc[item]
        else:
            hist = None

        if item in self.composition_dict.keys():
            comp = self.composition_dict[item]
        else:
            comp = None

        core_out = {'histology': hist, 'composition': comp}

        if self.extra_dfs is not None:
            for extra_name in self.extra_dfs_names:
                if item in self.histology_df.index:
                    extra_val = self.extra_dfs[extra_name].loc[item]
                else:
                    extra_val = None

                core_out[extra_name] = extra_val

        return core_out

    def __len__(self):
        return len(self.common_ids)

    def subset_ids(self,
                   hist_condition: Optional[List[bool]] = None,
                   comp_condition: Optional[List[bool]] = None):

        if hist_condition is None:
            hist_condition = [True] * self.histology_df.shape[0]

        if comp_condition is None:
            comp_condition = [True] * len(self.composition_dict)

        list_ids = [list(self.histology_df.index.astype(str)[hist_condition]),
                    list(np.array(list(self.composition_dict.keys()))[comp_condition])]
        if self.extra_dfs is not None:
            for extra_name in self.extra_dfs_names:
                list_ids.append(self.extra_dfs[extra_name].index.astype(str))

        common_ids = list(set.intersection(*map(set, list_ids)))
        self.common_ids = common_ids

    def exist_comp_condition(self, th=-1):
        valid_ids = []
        for item in self.composition_dict.keys():
            if self.composition_dict[item].mean(axis=0)[:-1].sum() / self.composition_dict[item].mean(
                    axis=0).sum() > th:
                valid_ids.append(True)
            else:
                valid_ids.append(False)
        return valid_ids

    def filter_by_clone(self, clone_id, th=-1):
        valid_ids = []
        if type(clone_id) == list:
            for item in self.common_ids:
                logic = self.composition_dict[item].mean(axis=0)[:-1].argmax() == clone_id[0]
                for i in range(1, len(clone_id)):
                    logic |= self.composition_dict[item].mean(axis=0)[:-1].argmax() == clone_id[i]
                if self.composition_dict[item].mean(axis=0)[clone_id].sum() / self.composition_dict[item].mean(
                        axis=0).sum() > th and logic:
                    valid_ids.append(True)
                else:
                    valid_ids.append(False)
        else:
            for item in self.common_ids:
                if self.composition_dict[item].mean(axis=0)[clone_id] / self.composition_dict[item].mean(
                        axis=0).sum() > th and self.composition_dict[item].mean(axis=0)[:-1].argmax() == clone_id:
                    valid_ids.append(True)
                else:
                    valid_ids.append(False)
        return valid_ids

    def sort_ids(self, th=1):
        compos_list = [self.composition_dict[i].mean(axis=0) / self.composition_dict[i].mean(axis=0).sum() for i in
                       self.common_ids]
        sorting_ids = np.argsort(np.array(
            list(map(lambda x: (~(x[:-1].sum() > th), np.argmax(x[:-1]), -np.max(x[:-1])), compos_list)) + [None],
            dtype=tuple)[:-1])
        self.common_ids = list(np.array(self.common_ids)[sorting_ids])

    @property
    def comp_matrix(self):
        try:
            compos_array = np.stack([self.composition_dict[i] for i in self.common_ids]).mean(axis=1)
        except ValueError:
            compos_array = np.empty((0, self.n_subclones))
        return compos_array

    @property
    def hist_matrix(self):
        hist_matrix = self.histology_df.loc[self.common_ids, :]
        return hist_matrix

    def extra_matrix(self, name):
        extra_matrix = self.extra_dfs[name].loc[self.common_ids, :]
        return extra_matrix
