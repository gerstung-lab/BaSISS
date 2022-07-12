from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd


class Histogenomic_associations:

    """A structural class to store histological annotations, phylogenetic composition and extra information. Essentially
    merging several layers of data (BaSISS (mut), ISS expression (exp, imm) and histology

        Attributes
        ----------
        histology_df : pd.DataFrame
            Histological features DataFrame with region names as indexes
        composition_dict : dict
            Clonal composition dictionary, where key names correspond to regions names
        extra_dfs : dict
            Dictionary of extra DataFrame, indexes should correspond to region names
        n_subclones : int
            Number of clones
        common_ids : list
            List of region ids found in all layers of data
        hist_matrix : pd.DataFrame
            Histology features properly subset and ordered
        comp_matrix : np.array
            Clonal composition matrix properly subset and ordered

        Methods
        -------
        subset_ids(hist_condition: Optional[List[bool]] = None, comp_condition: Optional[List[bool]] = None):
            subset indices based on histological and clonal compositional conditions

        exist_comp_condition(th=-1):
            return regions' indices based on cancer cell fraction threshold

        filter_by_clone(clone_id, th=-1):
            return regions' indices based on clonal cell fraction threshold (clone specific)

        sort_ids(th=1):
            sorting regions indices based on clonal composition

        extra_matrix(name):
            returns extra feature DataFrame properly subset and ordered

        """
    def __init__(self,
                 histo_df: pd.DataFrame,
                 composition_dict: dict,
                 extra_dfs: Optional[Dict[str, pd.DataFrame]] = None
                 ):
        """Constructor

        Parameters
        ----------
        histo_df : pd.DataFrame
            Histological features DataFrame with region names as indexes
        composition_dict
            Clonal composition dictionary, where key names correspond to regions names
        extra_dfs
            Dictionary of extra DataFrame, indexes should correspond to region names
        """
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
        """

        Parameters
        ----------
        item : str
            Region id

        Returns
        -------
        dict
            {'histology': [], 'composition': []}
        """
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
        """

        Parameters
        ----------
        hist_condition: List[bool]
            Histological conditioning
        comp_condition : List[bool]
            Clonal composition conditioning

        Returns
        -------

        """

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
        """Return regions with appropriate CCF

        Parameters
        ----------
        th : float
            CCF threshold

        Returns
        -------
        list
            List of filtered regions names
        """
        valid_ids = []
        for item in self.composition_dict.keys():
            if self.composition_dict[item].mean(axis=0)[:-1].sum() / self.composition_dict[item].mean(
                    axis=0).sum() > th:
                valid_ids.append(True)
            else:
                valid_ids.append(False)
        return valid_ids

    def filter_by_clone(self, clone_id, th=-1):
        """Return regions with appropriate clone cell fraction

        Parameters
        ----------
        clone_id : List[int] or int
            Clones to consider
        th : float
            Clone cell fraction threshold
        Returns
        -------
        list
            List of filtered regions names
        """
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
        """Sorting of regions based on dominant clone fraction

        Parameters
        ----------
        th : float
            Clone cell fraction (among cancer clones)

        Returns
        -------

        """
        compos_list = [self.composition_dict[i].mean(axis=0) / self.composition_dict[i].mean(axis=0).sum() for i in
                       self.common_ids]
        sorting_ids = np.argsort(np.array(
            list(map(lambda x: (~(x[:-1].sum() > th), np.argmax(x[:-1]), -np.max(x[:-1])), compos_list)) + [None],
            dtype=tuple)[:-1])
        self.common_ids = list(np.array(self.common_ids)[sorting_ids])

    @property
    def comp_matrix(self):
        """

        Returns
        -------
        np.array
            Clonal composition information matrix properly subset and ordered
        """
        try:
            compos_array = np.stack([self.composition_dict[i] for i in self.common_ids]).mean(axis=1)
        except ValueError:
            compos_array = np.empty((0, self.n_subclones))
        return compos_array

    @property
    def hist_matrix(self):
        """

        Returns
        -------
        pd.DataFrame
            Histological information DataFrame properly subset and ordered
        """
        hist_matrix = self.histology_df.loc[self.common_ids, :]
        return hist_matrix

    def extra_matrix(self, name):
        """

        Parameters
        ----------
        name : str
            Name of the extra data layer

        Returns
        -------
        pd.DataFrame
            Extra information DataFrame properly subset and ordered
        """
        extra_matrix = self.extra_dfs[name].loc[self.common_ids, :]
        return extra_matrix
