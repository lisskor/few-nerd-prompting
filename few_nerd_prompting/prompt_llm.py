import argparse
import logging

from read_few_nerd import FewNerdEpisodesSet
from clarifai_prompter import ClarifaiPrompter
from prompt_building_utils import build_llama2_prompt_plain, build_self_verification_prompt_plain
from prompt_building_utils import extract_predicted_entities

logging.basicConfig(format="{asctime} {levelname}: {message}",
                    style="{", level=logging.INFO)


def main(args):
    all_episodes = FewNerdEpisodesSet(args.data_file)

    system_message = f"I am an excellent linguist. The task is to label {args.entity_class} entities " \
                     "in the given sentence. Below are some examples:"
    system_message_verification = f"The task is to verify whether the word is a " \
                                  f"{args.entity_class} entity extracted from the given sentence"
    prompter = ClarifaiPrompter(args.user_id, args.app_id, args.pat)

    for episode in all_episodes.episodes:
        episode.gpt_ner_examples_from_episode(args.entity_class)

        raw_texts_ner = [
            build_llama2_prompt_plain(few_shot_examples=zip(episode.support_input_examples,
                                                            episode.support_output_examples),
                                      system_msg=system_message,
                                      input_example=query_input_example)
            for query_input_example in episode.query_input_examples
        ]
        outputs = prompter.predict(args.model_id, raw_texts_ner)

        for query_input, raw_text, output, correct_output in zip(
                episode.query_input_examples, raw_texts_ner, outputs, episode.query_output_examples
        ):
            logging.info("PROMPT:\n")
            logging.info(raw_text + '\n')

            logging.info("OUTPUT:\n")
            logging.info(output + "\n")

            logging.info("CORRECT OUTPUT:\n")
            logging.info(correct_output + '\n')

            extracted_entities = extract_predicted_entities(output.split("\n")[0])
            logging.info(f"PREDICTED ENTITIES: {extracted_entities}\n")

            if extracted_entities:
                raw_texts_verification = [build_self_verification_prompt_plain(system_msg=system_message_verification,
                                                                               input_example=query_input,
                                                                               candidate_entity=candidate,
                                                                               entity_class=args.entity_class)
                                          for candidate in extracted_entities]

                verification_outputs = prompter.predict(args.model_id, raw_texts_verification)
                for raw_text_verification, verification_output in zip(raw_texts_verification, verification_outputs):
                    logging.info("VERIFICATION PROMPT:\n")
                    logging.info(raw_text_verification + '\n')

                    logging.info("VERIFICATION OUTPUT:\n")
                    logging.info(verification_output + "\n")


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
        '-c', '--entity_class',
        default='event',
        type=str,
        help='Entity class for current demonstration & input.'
    )

    arguments = parser.parse_args()
    main(arguments)
