import argparse
import logging
import json

from read_few_nerd import FewNerdEpisodesSet
from clarifai_prompter import ClarifaiPrompter
from prompt_building_utils import build_llama2_prompt_plain, build_self_verification_prompt_plain, labels_from_output
from prompt_building_utils import extract_predicted_entities

logging.basicConfig(format="{asctime} {levelname}: {message}",
                    style="{", level=logging.INFO)


def main(args):
    all_episodes = FewNerdEpisodesSet(args.data_file)

    system_messages = {entity_class: f"I am an excellent linguist. The task is to label {entity_class} entities "
                       "in the given sentence. Below are some examples:" for entity_class in args.entity_classes}
    system_messages_verification = {entity_class: f"The task is to verify whether the word is a "
                                    f"{entity_class} entity extracted from the given sentence"
                                    for entity_class in args.entity_classes}
    prompter = ClarifaiPrompter(args.user_id, args.app_id, args.pat)

    with open(args.output_file, 'w', encoding='utf8') as out_fh:

        episode_counter = 0
        for episode in all_episodes.episodes:
            if episode_counter > args.max_episodes:
                break
            result = {"text": {entity_class: [] for entity_class in args.entity_classes},
                      "label": {entity_class: [] for entity_class in args.entity_classes}}
            logging.info(f"Episode {episode_counter}")

            for entity_class in args.entity_classes:
                logging.info(f"Class: {entity_class}")
                episode.gpt_ner_examples_from_episode(entity_class)

                for query_input_example, input_tokens, correct_output in zip(
                        episode.query_input_examples, episode.query_tokens, episode.query_output_examples
                ):
                    raw_text_ner = build_llama2_prompt_plain(few_shot_examples=zip(episode.support_input_examples,
                                                                                   episode.support_output_examples),
                                                             system_msg=system_messages[entity_class],
                                                             input_example=query_input_example)
                    logging.debug("PROMPT:\n")
                    logging.debug(raw_text_ner + '\n')
                    output = prompter.predict(args.model_id, [raw_text_ner])[0]
                    output_first_line = output.split('\n')[0]

                    logging.info(f"OUTPUT (1st line): {output_first_line}")
                    logging.info(f"CORRECT OUTPUT: {correct_output}")

                    result["text"][entity_class].append(output_first_line)
                    result["label"][entity_class].append(labels_from_output(output_first_line,
                                                                            input_tokens,
                                                                            entity_class))

                    extracted_entities = extract_predicted_entities(output_first_line)
                    logging.info(f"PREDICTED ENTITIES: {extracted_entities}")

                    if args.verification:
                        if extracted_entities:
                            raw_texts_verification = [build_self_verification_prompt_plain(
                                system_msg=system_messages_verification[entity_class],
                                input_example=query_input_example,
                                candidate_entity=candidate,
                                entity_class=entity_class
                            )
                                for candidate in extracted_entities]

                            for verification_prompt in raw_texts_verification:
                                verification_output = prompter.predict(args.model_id, [verification_prompt])[0]
                                logging.debug("VERIFICATION PROMPT:\n")
                                logging.debug(verification_prompt + '\n')
                                logging.info(f"VERIFICATION OUTPUT: {verification_output}")

            out_fh.write(json.dumps(result) + "\n")
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
        '--max_episodes',
        default=10,
        type=int,
        help='Max episodes to process (primarily for testing purposes)'
    )
    parser.add_argument(
        '--verification',
        action='store_true',
        help='Use self-verification'
    )

    arguments = parser.parse_args()
    main(arguments)
