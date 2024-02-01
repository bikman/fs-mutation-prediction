"""
This module contains definition of episode class and creation of episodes
"""
import random
from utils import CFG, normalize_list
from torch.utils.data import Dataset
from run_prism_data_creation import log
from torch.utils.data import DataLoader


class EpisodeDataset(Dataset):
    def __init__(self, protein_name, file_name, data):
        self.data = data
        self.protein_name = protein_name
        self.file_name = file_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        return data_item

    def __repr__(self):
        return f'{self.protein_name}:{self.__class__}:size={self.__len__()}'


class Episode(object):
    """
    Contains support and query sets
    Used to run a single cycle of train/test/validation on the model
    """

    def __init__(self):
        """
        Ctor
        """
        self.support_dss = []
        self.query_dss = []

    def __repr__(self):
        res = 'Episode proteins: '
        for ds in self.support_dss:
            res += f'{ds.protein_name}, '
        return res[:-2]


class TestEpisode(object):
    def __init__(self):
        """
        Episode used for testing (evaluation) of the model
        """
        self.support_ds = None
        self.query_ds = None

    def __repr__(self):
        return f'Test Episode protein: {self.support_ds.protein_name}'


class EpisodesCreator(object):
    """
    Creates an instance of Episode class
    Uses 'num_ways' of DIFFERENT proteins
    [protein_id | position | ddE | ddG | src | dst | diff emb | score]
    """

    def __init__(self, dss, eval_prot_filename):
        """
        ctor
        @param dss:
        @param eval_prot_filename: protein file to be used for validation (if enabled in CFG)
        """
        self.eval_prot_filename = eval_prot_filename
        self.dss = dss
        self.valid_prot_filename = random.choice([x.file_name for x in dss if x.file_name != eval_prot_filename])

    def normalize_variants_scores(self, variants):
        """
        Normalize the scores
        @param variants: list of 8 values used as input for model
        """
        normalize_episode = int(CFG['episodes_train']['norm_episode'])
        log(f'{normalize_episode=}')
        if normalize_episode == 1:
            self.normalize_variants_scores_aux(variants)

    def normalize_variants_scores_aux(self, variants):
        """
        Normalize the scores (6th element in list)
        @param variants: list of 8 values used as input for model
        """
        score_ndx = 6  # index of score in variant tensor
        values = [v[score_ndx] for v in variants]
        norm_values = normalize_list(values)
        assert len(norm_values) == len(variants)
        for idx, variant in enumerate(variants):
            variants[idx][score_ndx] = norm_values[idx]

    def create_train_episode(self, num_ways, support_set_size, query_set_size):
        """
        TBD
        @param num_ways:
        @param support_set_size:
        @param query_set_size:
        @return:
        """
        if int(CFG['episodes_train']['use_validation']) == 1:
            # validation
            train_dss = [x for x in self.dss if
                         x.file_name != self.eval_prot_filename and x.file_name != self.valid_prot_filename]
            assert len(train_dss) == len(self.dss) - 2
        else:
            # no validation
            train_dss = [x for x in self.dss if x.file_name != self.eval_prot_filename]
            assert len(train_dss) == len(self.dss) - 1

        selected_dss = random.sample(train_dss, num_ways)
        assert len(selected_dss) == num_ways
        episode = Episode()
        for ds in selected_dss:
            total_variants_num = support_set_size + query_set_size
            if total_variants_num >= len(ds.data):
                raise Exception(f'{ds.protein_name}: {total_variants_num} is bigger than size of ds {len(ds.data)}')
            selected_variants = random.sample(ds.data, total_variants_num)
            support_variants = selected_variants[:support_set_size]
            query_variants = selected_variants[support_set_size:]
            log(f'Normalizing scores for train episode: {ds.protein_name}')
            self.normalize_variants_scores_aux(support_variants)
            self.normalize_variants_scores_aux(query_variants)
            # self.normalize_variants_scores(support_variants)
            # self.normalize_variants_scores(query_variants)
            assert len(support_variants) == support_set_size
            assert len(query_variants) == query_set_size
            support_ds = EpisodeDataset(ds.protein_name, ds.file_name, support_variants)
            query_ds = EpisodeDataset(ds.protein_name, ds.file_name, query_variants)
            episode.support_dss.append(support_ds)
            episode.query_dss.append(query_ds)
        assert len(episode.support_dss) == len(episode.query_dss)
        assert len(episode.support_dss) == num_ways
        return episode

    def create_validation_episode(self, support_set_size):
        """
        TBD
        @return:
        """
        valid_dss = [x for x in self.dss if x.file_name == self.valid_prot_filename]
        assert len(valid_dss) == 1
        valid_ds = valid_dss[0]
        episode = TestEpisode()
        random.shuffle(valid_ds.data)
        set_size = support_set_size
        support_variants = valid_ds.data[:set_size]
        query_variants = valid_ds.data[set_size:]
        log('Normalizing scores for valid episode')
        self.normalize_variants_scores_aux(support_variants)
        self.normalize_variants_scores_aux(query_variants)
        # self.normalize_variants_scores(support_variants)
        # self.normalize_variants_scores(query_variants)
        # log('*' * 30)
        # log(f'Support validation set size={len(support_variants)}')
        # log(f'Query validation set size={len(query_variants)}')
        # log('*' * 30)
        episode.support_ds = EpisodeDataset(valid_ds.protein_name, valid_ds.file_name, support_variants)
        episode.query_ds = EpisodeDataset(valid_ds.protein_name, valid_ds.file_name, query_variants)
        assert len(episode.query_ds) + len(episode.support_ds) == len(valid_ds)
        return episode

    def create_test_episode(self, support_set_size):
        """
        TBD
        @param support_set_size:
        @return:
        """
        test_dss = [x for x in self.dss if x.file_name == self.eval_prot_filename]
        assert len(test_dss) == 1
        test_ds = test_dss[0]
        episode = TestEpisode()
        random.shuffle(test_ds.data)
        set_size = support_set_size
        support_variants = test_ds.data[:set_size]
        query_variants = test_ds.data[set_size:]
        log('Normalizing scores for test episode')
        self.normalize_variants_scores_aux(support_variants)
        self.normalize_variants_scores_aux(query_variants)
        log('*' * 30)
        log(f'Support test set size={len(support_variants)}')
        log(f'Query test set size={len(query_variants)}')
        log('*' * 30)
        episode.support_ds = EpisodeDataset(test_ds.protein_name, test_ds.file_name, support_variants)
        episode.query_ds = EpisodeDataset(test_ds.protein_name, test_ds.file_name, query_variants)
        assert len(episode.query_ds) + len(episode.support_ds) == len(test_ds)
        return episode


class DataContext(object):
    def __init__(self):
        self.support_loaders_list = []
        self.query_loaders_list = []

    def fill_data_loaders(self, episode):
        batch_size = int(CFG['flow_train']['batch_size'])
        supp_loaders = []
        query_loaders = []
        for supp_ds in episode.support_dss:
            supp_loader = DataLoader(supp_ds, batch_size=batch_size, shuffle=True)
            supp_loaders.append(supp_loader)
        for query_ds in episode.query_dss:
            query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=True)
            query_loaders.append(query_loader)
        self.support_loaders_list = supp_loaders
        self.query_loaders_list = query_loaders

    def fill_test_data_loaders(self, test_episode):
        batch_size = int(CFG['flow_train']['batch_size'])
        self.support_loaders_list = [DataLoader(test_episode.support_ds, batch_size=batch_size, shuffle=True)]
        self.query_loaders_list = [DataLoader(test_episode.query_ds, batch_size=batch_size, shuffle=True)]


if __name__ == '__main__':
    pass
