import argparse
import json
import logging
from typing import List, Dict, Optional

from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report

from read_few_nerd import FewNerdEpisodesSet

logging.basicConfig(format="{asctime} {levelname}: {message}",
                    style="{", level=logging.INFO)


def coarse_grained_from_fine_grained(labels: List[str]) -> List[str]:
    """
    Transform fine-grained Few-NERD labels into coarse-grained, e.g.
    ["art-music", "art-music", "O", "O", "building-theater"] -> ["art", "art", "O", "O", "building"]

    :param labels: list of fine-grained token labels
    :return: list of coarse-grained token labels
    """
    return [label if label == "O" else label.split("-")[0] for label in labels]


def single_class(labels: List[str], entity_class: str) -> List[str]:
    """
    Given a list of token class labels, keep only a single class as positive, replacing all other class labels with "O"
    E.g. with entity_class = "building"
    ["art", "art", "O", "O", "building"] -> ["O", "O", "O", "O", "building"]

    :param labels: list of token labels
    :param entity_class: class to keep
    :return: list of token labels with only the entity_class preserved
    """
    return [label if label == entity_class else "O" for label in labels]


def labels_to_iob(labels: List[str]) -> List[str]:
    """
    Transform labels to IOB format, e.g.
    ["art", "art", "O", "O", "building"] -> ["B-art", "I-art", "O", "O", "B-building"]
    The transformation may be incorrect if there are two adjacent entities of the same type,
    as there is no way to distinguish between them.

    :param labels: list of token labels
    :return: list of token labels in IOB format
    """
    result = []
    chunk_started = False
    for label in labels:
        if label == 'O':
            result.append(label)
            chunk_started = False
        elif label != 'O' and not chunk_started:
            result.append(f"B-{label}")
            chunk_started = True
        elif label != '0' and chunk_started:
            result.append(f"I-{label}")
    return result


def get_true_labels(filename: str, entity_class: str, full_labels_path: Optional[str], pred_ids: List[int] = None,
                    coarse_grained: bool = True, full_labels: bool = False) -> Dict[int, List[List[str]]]:
    """
    Get ground truth labels from a Few-NERD episode data file, taking into account only one entity type.

    :param filename: path to ground truth Few-NERD episode file
    :param entity_class: entity class to keep (all others will be replaced with "O")
    :param full_labels_path: path to files containing labels from the supervised task
    :param pred_ids: IDs of episodes for which predictions are present (in case some predictions are missing)
    :param coarse_grained: if true, use coarse-grained classes
    :param full_labels: use full labels from the supervised task
    :return: dictionary mapping episode IDs to token labels for each sentence of the query set
    """
    all_episodes = FewNerdEpisodesSet(filename=filename, full_labels_path=full_labels_path, full_labels=full_labels)
    all_true_labels = {}

    episode_id = 0
    for episode in all_episodes.episodes:
        if episode_id in pred_ids:
            all_true_labels[episode_id] = []
            episode.gpt_ner_examples_from_episode(entity_class)
            if coarse_grained:
                all_true_labels[episode_id].extend([labels_to_iob(single_class(coarse_grained_from_fine_grained(query),
                                                                               entity_class))
                                                    for query in episode.query_labels])
            else:
                all_true_labels[episode_id].extend(episode.query_labels)

        episode_id += 1

    return all_true_labels


def read_predicted_labels_single_class(filename: str, entity_class: str) -> Dict[int, List[List[str]]]:
    """
    Get predicted labels from a file generated with prompt_llm.py, taking into account only one entity type.

    :param filename: path to file containing predictions
    :param entity_class: entity class to keep (all others will be replaced with "O")
    :return: dictionary mapping episode IDs to predicted token labels for each sentence of the query set
    """
    all_predicted_labels = {}
    with open(filename, 'r', encoding="utf8") as fh:
        for line in fh.readlines():
            prediction = json.loads(line.strip())
            episode_id = list(prediction.keys())[0]
            all_predicted_labels.setdefault(int(episode_id),
                                            [labels_to_iob(p) for p in prediction[episode_id]['label'][entity_class]])
    return all_predicted_labels


def build_report(entity_classes: List[str], pred_file: str, true_file: str, round_to: int,
                 full_labels: bool, full_labels_path: Optional[str]) -> str:
    """
    Iterate over all required entity classes and build a table of metrics x entity classes
    (showing precision, recall, F1-score, and support for each class).

    :param entity_classes: all entity classes to calculate scores for
    :param pred_file: path to file containing model predictions (generated by prompt_llm.py)
    :param true_file: path to Few-NERD episode data file with ground truth labels
    :param round_to: max decimal places for metrics
    :param full_labels: use full labels from the supervised task
    :param full_labels_path: path to files containing labels from the supervised task
    :return: string showing a table of metrics x entity classes
    """
    result = ""

    for entity_class in entity_classes:
        pred_labels = read_predicted_labels_single_class(pred_file, entity_class)
        true_labels = get_true_labels(filename=true_file, entity_class=entity_class, full_labels_path=full_labels_path,
                                      pred_ids=[key for key, value in sorted(pred_labels.items())],
                                      full_labels=full_labels)

        # Transform predicted and true labels from dicts into lists
        sorted_true_list = [value for key, value in sorted(true_labels.items())]
        sorted_pred_list = [value for key, value in sorted(pred_labels.items())]
        true_flattened = [item for sublist in sorted_true_list for item in sublist]
        pred_flattened = [item for sublist in sorted_pred_list for item in sublist]

        # Temporary fix: fill the labels that errored out with O's
        for i in range(len(true_flattened)):
            if not pred_flattened[i]:
                pred_flattened[i] = ["O"] * len(true_flattened[i])

        class_report = report(true_flattened, pred_flattened, round_to)

        if not result:
            result += class_report.split("\n")[0]
            result += "\n\n"

        result += class_report.split("\n")[2]
        result += "\n"

    return result


def report(y_true: List[List[str]], y_pred: List[List[str]], round_to) -> str:
    """
    Create a classification report using seqeval.

    :param y_true: true labels
    :param y_pred: predicted labels
    :param round_to: max decimal places for scores
    :return: classification_report generated by seqeval
    """
    assert([len(s) for s in y_true] == [len(s) for s in y_pred]), "Token counts do not match"
    return classification_report(y_true, y_pred, digits=round_to)


def main(args):
    scores = build_report(entity_classes=args.entity_classes, pred_file=args.pred_file,
                          true_file=args.few_nerd_file, round_to=args.decimal_places,
                          full_labels=args.full_labels, full_labels_path=args.full_labels_data_path)
    logging.info(f"Scores:\n{scores}")
    # print(scores)


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
    parser.add_argument(
        '-d', '--decimal_places',
        default=3,
        type=int,
        help='Round scores in output to N decimal places.'
    )
    parser.add_argument(
        '--full_labels',
        default=False,
        action='store_true',
        help='Use full labels from the supervised task.'
    )
    parser.add_argument(
        '-f', '--full_labels_data_path',
        type=str,
        help='Path to files with full data labels.'
    )

    arguments = parser.parse_args()
    main(arguments)
