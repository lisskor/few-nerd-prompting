import os
import json

from typing import Generator, Dict

from prompt_building_utils import make_output_example


class FewNerdEpisode:
    def __init__(self, episode_dict: Dict):
        self.support_set = episode_dict['support']
        self.query_set = episode_dict['query']

        self.query_tokens = self.query_set['word']
        self.query_labels = self.query_set['label']

        self.support_input_examples, self.support_output_examples = None, None
        self.query_input_examples, self.query_output_examples = None, None

    def gpt_ner_examples_from_episode(self, entity_class: str):
        self.support_input_examples = [' '.join(sentence) for sentence in self.support_set['word']]
        self.support_output_examples = [make_output_example(sentence, labels, entity_class)
                                        for sentence, labels in
                                        zip(self.support_set['word'], self.support_set['label'])]

        self.query_input_examples = [' '.join(sentence) for sentence in self.query_set['word']]
        self.query_output_examples = [make_output_example(sentence, labels, entity_class)
                                      for sentence, labels in zip(self.query_set['word'], self.query_set['label'])]

class FewNerdEpisodesSet:
    def __init__(self, filename):
        self.filename = filename
        # The below code assumes file names as in the original FewNERD structure,
        # e.g. Few-NERD/data/episode-data/intra/dev_5_1.jsonl for 5-way 1~2-shot
        # train / dev / test
        self.split = os.path.basename(filename).split("_")[0]
        # inter / intra
        self.task_type = os.path.normpath(filename).split(os.path.sep)[-2]
        # N-way (5 / 10)
        self.n_way = os.path.basename(filename).split("_")[1]
        # N-shot (1~2 / 5~10)
        self.n_shot = os.path.basename(filename).split("_")[2].split(".")[0]

        self.episodes = self.read_file()

    def read_file(self) -> Generator[FewNerdEpisode]:
        with open(self.filename, 'r', encoding='utf8') as json_file:
            for line in json_file:
                episode = FewNerdEpisode(json.loads(line.strip()))
                yield episode
