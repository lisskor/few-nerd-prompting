import argparse
import logging
import json

# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm

from read_few_nerd import FewNerdEpisodesSet
from clarifai_prompter import ClarifaiPrompter
from prompt_building_utils import build_self_verification_prompt_plain, extract_predicted_entities

logging.basicConfig(format="{asctime} {levelname}: {message}",
                    style="{", level=logging.INFO)


def read_input_file(filename):
    input_episodes = []
    with open(filename, 'r', encoding="utf8") as fh:
        for line in fh.readlines():
            input_episodes.append(json.loads(line.strip()))
    return input_episodes


def main(args):
    raw_episodes = FewNerdEpisodesSet(args.raw_data_file).episodes
    pred_episodes = read_input_file(args.pred_data_file)

    # Predicted classes should be the same for each episode
    entity_classes = list(pred_episodes[0]["text"].keys())

    system_messages_verification = {entity_class: f"The task is to verify whether the word is a "
                                    f"{entity_class} entity extracted from the given sentence"
                                    for entity_class in entity_classes}
    prompter = ClarifaiPrompter(args.user_id, args.app_id, args.pat)

    with open(args.output_file, 'w', encoding='utf8') as out_fh:

        episode_counter = 0
        for raw_episode, pred_episode in zip(raw_episodes, pred_episodes):
            if episode_counter > args.max_episodes:
                break
            logging.info(f"Episode {episode_counter}")

            for entity_class in entity_classes:
                logging.info(f"Class: {entity_class}")

                raw_texts = raw_episode.query_input_examples

                pred_texts = pred_episode["text"][entity_class]

                for raw_text, pred_text in zip(raw_texts, pred_texts):
                    extracted_entities = extract_predicted_entities(pred_text)
                    if extracted_entities:
                        raw_texts_verification = [build_self_verification_prompt_plain(
                            system_msg=system_messages_verification[entity_class],
                            input_example=raw_text,
                            candidate_entity=candidate,
                            entity_class=entity_class
                        )
                            for candidate in extracted_entities]

                        for verification_prompt in raw_texts_verification:
                            verification_output = prompter.predict(args.model_id, verification_prompt, 0)[0]
                            logging.info("VERIFICATION PROMPT:\n")
                            logging.info(verification_prompt + '\n')
                            logging.info(f"VERIFICATION OUTPUT: {verification_output}")

            # out_fh.write(json.dumps(results) + "\n")
            episode_counter += 1


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
        '-r', '--raw_data_file',
        type=str,
        help='File with raw episode data.'
    )
    parser.add_argument(
        '-p', '--pred_data_file',
        type=str,
        help='File with entity predictions for episodes.'
    )
    parser.add_argument(
        '-o', '--output_file',
        default='test.out',
        type=str,
        help='Output file.'
    )
    parser.add_argument(
        '--max_episodes',
        default=10,
        type=int,
        help='Max episodes to process (primarily for testing purposes)'
    )

    arguments = parser.parse_args()
    main(arguments)
