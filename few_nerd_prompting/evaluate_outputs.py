import argparse
import json
import logging
from typing import List, Dict

from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report

from read_few_nerd import FewNerdEpisodesSet

logging.basicConfig(format="{asctime} {levelname}: {message}",
                    style="{", level=logging.INFO)


def coarse_grained_from_fine_grained(labels: List[str]) -> List[str]:
    return [label if label == "O" else label.split("-")[0] for label in labels]


def single_class(labels: List[str], entity_class: str) -> List[str]:
    return [label if label == entity_class else "O" for label in labels]


def get_true_labels(filename: str, entity_class: str, pred_ids: List[int] = None, coarse_grained: bool = True) -> Dict[int, List[List[str]]]:
    all_episodes = FewNerdEpisodesSet(filename)
    all_true_labels = {}

    episode_id = 0
    for episode in all_episodes.episodes:
        if episode_id in pred_ids:
            all_true_labels[episode_id] = []
            episode.gpt_ner_examples_from_episode(entity_class)
            if coarse_grained:
                all_true_labels[episode_id].extend([single_class(coarse_grained_from_fine_grained(query), entity_class)
                                                    for query in episode.query_labels])
            else:
                all_true_labels[episode_id].extend(episode.query_labels)

        episode_id += 1

    return all_true_labels


def read_predicted_labels_single_class(filename: str, entity_class: str) -> Dict[int, List[List[str]]]:
    all_predicted_labels = {}
    with open(filename, 'r', encoding="utf8") as fh:
        for line in fh.readlines():
            prediction = json.loads(line.strip())
            episode_id = list(prediction.keys())[0]
            all_predicted_labels.setdefault(int(episode_id), prediction[episode_id]['label'][entity_class])
    return all_predicted_labels


def report(y_true: List[List[str]], y_pred: List[List[str]]):
    assert([len(s) for s in y_true] == [len(s) for s in y_pred]), "Token counts do not match"

    print(f"ACC: {accuracy_score(y_true, y_pred)}")
    print(f"PREC: {precision_score(y_true, y_pred)}")
    print(f"REC: {recall_score(y_true, y_pred)}")
    print(f"F1: {f1_score(y_true, y_pred)}")
    # print(classification_report(y_true, y_pred))


def main(args):
    for entity_class in args.entity_classes:
        pred_labels = read_predicted_labels_single_class(args.pred_file, entity_class)
        true_labels = get_true_labels(args.few_nerd_file, entity_class, [key for key, value in sorted(pred_labels.items())])

        logging.info(f"Class: {entity_class}")

        # Transform predicted and true labels from dicts into lists
        sorted_true_list = [value for key, value in sorted(true_labels.items())]
        sorted_pred_list = [value for key, value in sorted(pred_labels.items())]
        true_flattened = [item for sublist in sorted_true_list for item in sublist]
        pred_flattened = [item for sublist in sorted_pred_list for item in sublist]

        # Temporary fix: fill the labels that errored out with O's
        for i in range(len(true_flattened)):
            if not pred_flattened[i]:
                pred_flattened[i] = ["O"] * len(true_flattened[i])

        report(true_flattened, pred_flattened)


if __name__ == '__main__':
    # Add arguments to argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--few_nerd_file',
        type=str,
        help='Path to file with episode data and true labels.')
    parser.add_argument(
        '--pred_file',
        type=str,
        help='File with LLM predictions.'
    )
    parser.add_argument(
        '-c', '--entity_classes',
        default=['event'],
        type=str,
        nargs='+',
        help='Entity classes for current demonstration & input.'
    )

    arguments = parser.parse_args()
    main(arguments)
