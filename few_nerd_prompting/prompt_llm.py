import argparse
import logging
import json

from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from read_few_nerd import FewNerdEpisodesSet
from clarifai_prompter import ClarifaiPrompter
from prompt_building_utils import build_llama2_prompt_plain, labels_from_output

logging.basicConfig(format="{asctime} {levelname}: {message}",
                    style="{", level=logging.INFO)


def main(args):
    all_episodes = FewNerdEpisodesSet(args.data_file)

    system_messages = {entity_class: f"I am an excellent linguist. The task is to label {entity_class} entities "
                       "in the given sentence. Below are some examples:" for entity_class in args.entity_classes}
    prompter = ClarifaiPrompter(args.user_id, args.app_id, args.pat, args.max_tokens)

    last_episode_id = args.first_episode + args.n_episodes if args.n_episodes else None

    with open(args.output_file, 'w', encoding='utf8') as out_fh:
        episode_id = args.first_episode
        for episode in islice(all_episodes.episodes, args.first_episode, last_episode_id):
            results = {episode_id: {"text": {entity_class: [] for entity_class in args.entity_classes},
                                    "label": {entity_class: [] for entity_class in args.entity_classes}}}
            logging.info(f"Episode {episode_id}")

            raw_texts_ner = []
            for entity_class in args.entity_classes:
                episode.gpt_ner_examples_from_episode(entity_class)
                raw_texts_ner.extend([
                    (build_llama2_prompt_plain(few_shot_examples=zip(episode.support_input_examples,
                                                                     episode.support_output_examples),
                                               system_msg=system_messages[entity_class],
                                               input_example=query_input_example),
                     (episode_id, entity_class, i))
                    for i, query_input_example in enumerate(episode.query_input_examples)
                ])

            threads = []
            output_first_lines = {entity_class: [] for entity_class in args.entity_classes}

            with ThreadPoolExecutor(max_workers=10) as executor:
                for raw_text, query_index in tqdm(
                    raw_texts_ner,
                    total=len(episode.query_input_examples) * len(args.entity_classes),
                    desc="Getting predictions (submit)"
                ):
                    threads.append(executor.submit(prompter.predict, args.model_id, raw_text, query_index))

                for task in tqdm(as_completed(threads), total=len(raw_texts_ner), desc='Getting predictions (results)'):
                    result_text, (received_episode_id, entity_class, query_id) = task.result()
                    output_first_lines[entity_class].append((result_text.split('\n')[0], query_id))

            for entity_class in args.entity_classes:
                episode.gpt_ner_examples_from_episode(entity_class)
                results[episode_id]["text"][entity_class] = [t[0] for t
                                                             in sorted(output_first_lines[entity_class],
                                                                       key=lambda x: x[1])]
                for output, input_tokens in zip(results[episode_id]["text"][entity_class],
                                                episode.query_tokens):
                    try:
                        class_labels = labels_from_output(output, input_tokens, entity_class)
                    except IndexError:
                        class_labels = []
                    results[episode_id]["label"][entity_class].append(class_labels)

                logging.info(f"CLASS: {entity_class}")
                logging.info(f"OUTPUT (1st lines): {results[episode_id]['text'][entity_class]}")
                logging.info(f"CORRECT OUTPUT: {episode.query_output_examples}")

            out_fh.write(json.dumps(results) + "\n")
            episode_id += 1


if __name__ == '__main__':
    # Add arguments to argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--pat',
        type=str,
        help='Personal Access Token.')
    parser.add_argument(
        '-u', '--user_id',
        default='meta',
        type=str,
        help='User ID.')
    parser.add_argument(
        '-a', '--app_id',
        default="Llama-2",
        help='App ID.')
    parser.add_argument(
        '-m', '--model_id',
        default='llama2-7b-chat',
        type=str,
        help='Model ID.'
    )
    parser.add_argument(
        '-d', '--data_file',
        type=str,
        help='File with episode data.'
    )
    parser.add_argument(
        '-c', '--entity_classes',
        default=['event'],
        type=str,
        nargs='+',
        help='Entity classes for current demonstration & input.'
    )
    parser.add_argument(
        '-o', '--output_file',
        default='test.out',
        type=str,
        help='Output file.'
    )
    parser.add_argument(
        '--first_episode',
        type=int,
        default=0,
        help='Index of the first episode to predict (0-based)'
    )
    parser.add_argument(
        '--n_episodes',
        type=int,
        default=None,
        help='Number of episodes to predict'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=100,
        help="Max number of tokens to generate"
    )

    arguments = parser.parse_args()
    main(arguments)
