import argparse
import json
import logging
from typing import List

from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report

from read_few_nerd import FewNerdEpisodesSet

logging.basicConfig(format="{asctime} {levelname}: {message}",
                    style="{", level=logging.INFO)


def coarse_grained_from_fine_grained(labels: List[str]) -> List[str]:
    return [label if label == "O" else label.split("-")[0] for label in labels]


def single_class(labels: List[str], entity_class: str) -> List[str]:
    return [label if label == entity_class else "O" for label in labels]


def get_true_labels(filename: str, entity_class: str, max_episodes: int = None, coarse_grained: bool = True) -> List[List[List[str]]]:
    all_episodes = FewNerdEpisodesSet(filename)
    all_true_labels = []

    episode_counter = 0
    for episode in all_episodes.episodes:
        if max_episodes and episode_counter >= max_episodes:
            break
        episode.gpt_ner_examples_from_episode(entity_class)
        if coarse_grained:
            all_true_labels.append([single_class(coarse_grained_from_fine_grained(query), entity_class)
                                    for query in episode.query_labels])
        else:
            all_true_labels.append(episode.query_labels)
        episode_counter += 1

    return all_true_labels


def read_predicted_labels_single_class(filename: str, entity_class: str) -> List[List[List[str]]]:
    all_predicted_labels = []
    with open(filename, 'r', encoding="utf8") as fh:
        for line in fh.readlines():
            all_predicted_labels.append(json.loads(line.strip())['label'][entity_class])
    return all_predicted_labels


def report(y_true: List[List[str]], y_pred: List[List[str]]):
    print(f"ACC: {accuracy_score(y_true, y_pred)}")
    print(f"PREC: {precision_score(y_true, y_pred)}")
    print(f"REC: {recall_score(y_true, y_pred)}")
    print(f"F1: {f1_score(y_true, y_pred)}")
    # print(classification_report(y_true, y_pred))


def main(args):
    for entity_class in args.entity_classes:
        pred_labels = read_predicted_labels_single_class(args.pred_file, entity_class)
        true_labels = get_true_labels(args.few_nerd_file, entity_class, len(pred_labels))

        logging.info(f"Class: {entity_class}")

        report([item for sublist in true_labels for item in sublist],
               [item for sublist in pred_labels for item in sublist])


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
