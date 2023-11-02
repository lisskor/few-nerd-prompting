import os
import json

from prompt_building_utils import make_output_example


class FewNerdEpisodesSet:
    def __init__(self, filename):
        self.filename = filename
        # train / dev / test
        self.split = os.path.basename(filename).split("_")[0]
        # inter / intra
        self.task_type = os.path.normpath(filename).split(os.path.sep)[-2]
        # N-way (5 / 10)
        self.n_way = os.path.basename(filename).split("_")[1]
        # N-shot (1~2 / 5~10)
        self.n_shot = os.path.basename(filename).split("_")[2].split(".")[0]

        self.episodes = self.read_file()

    def read_file(self):
        with open(self.filename, 'r', encoding='utf8') as json_file:
            episode_list = [FewNerdEpisode(json.loads(line.strip())) for line in json_file.readlines()]
        return episode_list


class FewNerdEpisode:
    def __init__(self, episode_dict):
        self.support_set = episode_dict['support']
        self.query_set = episode_dict['query']

        self.support_input_examples, self.support_output_examples = None, None
        self.query_input_examples, self.query_output_examples = None, None

    def gpt_ner_examples_from_episode(self, entity_class):
        self.support_input_examples = [' '.join(sentence) for sentence in self.support_set['word']]
        self.support_output_examples = [make_output_example(sentence, labels, entity_class)
                                        for sentence, labels in
                                        zip(self.support_set['word'], self.support_set['label'])]

        self.query_input_examples = [' '.join(sentence) for sentence in self.query_set['word']]
        self.query_output_examples = [make_output_example(sentence, labels, entity_class)
                                      for sentence, labels in zip(self.query_set['word'], self.query_set['label'])]
