import os
import json

from typing import Generator, Dict, List, Optional

from prompt_building_utils import make_output_example


def preprocess_file_to_dict(file_paths: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Process a list of text files where each file contains lines of tokens and their corresponding labels.
    Sentences are separated by blank lines. The function creates a dictionary where each key is a sentence
    and the value is another dictionary containing the tokens and their corresponding labels.

    :param file_paths: list of paths to the text files to be processed
    :return: dictionary with sentences as keys and inner dictionaries as values,
             each inner dictionary has two keys:
             - 'word': a list of tokens in the sentence,
             - 'label': a list of corresponding token labels
    """
    sentence_dict = {}
    current_words = []
    current_labels = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    if current_words:
                        sentence_str = ' '.join(current_words).lower()
                        sentence_dict[sentence_str] = {"word": current_words, "label": current_labels}
                        current_words = []
                        current_labels = []
                else:
                    token, tag = line.rsplit(maxsplit=1)
                    current_words.append(token.lower())
                    current_labels.append(tag)

            # Append the last sentence if the file doesn't end with a blank line
            if current_words:
                sentence_str = ' '.join(current_words)
                sentence_dict[sentence_str] = {"word": current_words, "label": current_labels}

    return sentence_dict


class FewNerdEpisode:
    def __init__(self, episode_dict: Dict,
                 full_labels_dict: Optional[Dict[str, Dict[str, List[str]]]],
                 full_labels: bool = True):
        self.support_set = episode_dict['support']
        self.query_set = episode_dict['query']

        self.query_tokens = self.query_set['word']

        self.support_input_examples = [' '.join(sentence) for sentence in self.support_set['word']]
        self.query_input_examples = [' '.join(sentence) for sentence in self.query_set['word']]
        self.support_output_examples, self.query_output_examples = None, None

        if full_labels:
            self.support_set['label'], self.query_set['label'] = [], []
            for sentence_text in self.support_input_examples:
                self.support_set['label'].append(full_labels_dict[sentence_text]['label'])
            for sentence_text in self.query_input_examples:
                self.query_set['label'].append(full_labels_dict[sentence_text]['label'])

        self.query_labels = self.query_set['label']

    def gpt_ner_examples_from_episode(self, entity_class: str):
        self.support_output_examples = [make_output_example(sentence, labels, entity_class)
                                        for sentence, labels in
                                        zip(self.support_set['word'], self.support_set['label'])]

        self.query_output_examples = [make_output_example(sentence, labels, entity_class)
                                      for sentence, labels in zip(self.query_set['word'], self.query_set['label'])]


class FewNerdEpisodesSet:
    def __init__(self, filename: str, full_labels_path: Optional[str], full_labels: bool = False):
        self.full_labels = full_labels
        if self.full_labels and not full_labels_path:
            raise Exception("File with full dataset labels not provided")
        all_label_files = [f"{full_labels_path}/{split}.txt" for split in ["train", "dev", "test"]]
        self.full_labels_dict = preprocess_file_to_dict(all_label_files) if full_labels else None

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

    def read_file(self) -> Generator[FewNerdEpisode, None, None]:
        with open(self.filename, 'r', encoding='utf8') as json_file:
            for line in json_file:
                episode = FewNerdEpisode(json.loads(line.strip()), self.full_labels_dict, self.full_labels)
                yield episode
