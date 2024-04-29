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
                all_true_labels[episode_id].extend([labels_to_bio(single_class(coarse_grained_from_fine_grained(query), entity_class))
                                                    for query in episode.query_labels])
            else:
                all_true_labels[episode_id].extend(episode.query_labels)

        episode_id += 1

    return all_true_labels


def labels_to_bio(labels: List[str]) -> List[str]:
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


def read_predicted_labels_single_class(filename: str, entity_class: str) -> Dict[int, List[List[str]]]:
    all_predicted_labels = {}
    with open(filename, 'r', encoding="utf8") as fh:
        for line in fh.readlines():
            prediction = json.loads(line.strip())
            episode_id = list(prediction.keys())[0]
            all_predicted_labels.setdefault(int(episode_id),
                                            [labels_to_bio(p) for p in prediction[episode_id]['label'][entity_class]])
    return all_predicted_labels


def build_report(entity_classes: List[str], pred_file: str, true_file: str, round_to: int) -> str:
    result = ""

    for entity_class in entity_classes:
        pred_labels = read_predicted_labels_single_class(pred_file, entity_class)
        true_labels = get_true_labels(true_file, entity_class, [key for key, value in sorted(pred_labels.items())])

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
    assert([len(s) for s in y_true] == [len(s) for s in y_pred]), "Token counts do not match"
    return classification_report(y_true, y_pred, digits=round_to)


def main(args):
    scores = build_report(entity_classes=args.entity_classes, pred_file=args.pred_file,
                          true_file=args.few_nerd_file, round_to=args.decimal_places)
    logging.info("Scores:")
    print(scores)


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

    arguments = parser.parse_args()
    main(arguments)
