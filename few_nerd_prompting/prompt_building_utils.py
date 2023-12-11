from typing import List, Tuple, Iterator

from nltk import word_tokenize

TAG_START = "@@"
TAG_END = "##"


def make_output_example(sentence: str, labels: List[str], entity_class: str) -> str:
    sentence_list = []
    # Keep track of whether we are currently writing an entity that needs to be marked
    entity_in_progress = False
    for w, l in zip(sentence, labels):
        if l.startswith(entity_class):
            # First token of entity: add with TAG_START
            if not entity_in_progress:
                sentence_list.append(f"{TAG_START}{w}")
                entity_in_progress = True
            # Second, third, etc. tokens of entity: add token only
            else:
                sentence_list.append(w)
        else:
            # If current entity just ended, modify the last added token to include TAG_END
            if entity_in_progress:
                sentence_list[-1] = f"{sentence_list[-1]}{TAG_END}"
                entity_in_progress = False
            # Then add current non-entity token
            sentence_list.append(w)
    return ' '.join(sentence_list)


def output_well_formed(sentence: str) -> bool:
    # Check that output is well-formed:
    # each TAG_START has a corresponding TAG_END,
    # the entities do not overlap (each one is closed before next is opened)

    # TODO: do we need to check for things like "@!"? How to find them?

    start, end = 0, len(sentence)
    start_pos = sentence.find(TAG_START, start, end)
    end_pos = sentence.find(TAG_END, start, end)

    # If there are no start or end tags found -> no entities -> good
    if start_pos == -1 and end_pos == -1:
        return True

    # End but no start -> bad
    if start_pos == -1 and end_pos != -1:
        return False

    # If there is a start tag
    while start_pos != -1:
        # End does not exist (-1) or comes before start -> bad
        if end_pos < start_pos:
            return False
        elif end_pos > start_pos:
            # Find next start
            next_start_pos = sentence.find(TAG_START, start_pos + 2, end)
            # No more starts
            if next_start_pos == -1:
                # Are there more ends?
                next_end_pos = sentence.find(TAG_END, end_pos + 2, end)
                # There is end after -> bad
                if next_end_pos != -1:
                    return False
                # No end after -> good
                elif next_end_pos == -1:
                    return True
            # If there are more starts
            else:
                # Next start comes before current end -> bad
                if end_pos > next_start_pos:
                    return False
                # If next start comes after current end, continue
                elif end_pos < next_start_pos:
                    start_pos = next_start_pos
                    end_pos = sentence.find(TAG_END, start_pos + 2, end)


def extract_predicted_entities(sentence: str) -> List[str]:
    predicted_entities = []
    start, end = 0, len(sentence)
    # Look for TAG_START
    entity_start = sentence.find(TAG_START, start, end)
    # If TAG_START found
    while entity_start != -1:
        # Move start of next search to the start of entity after TAG_START
        start = entity_start + 2
        # Look for TAG_END
        entity_end = sentence.find(TAG_END, start, end)
        # If TAG_END found
        if entity_end != -1:
            # Add found entity to list
            predicted_entities.append(sentence[start:entity_end])
        # Look for next entity
        entity_start = sentence.find(TAG_START, start, end)
    return predicted_entities


def build_llama2_prompt(few_shot_examples: List[str], system_msg: str, input_example: str) -> str:
    # Prompt with <s>[INST] <<SYS>> ... <</SYS>> ... [/INST]
    few_shot_examples_string = "\n".join([f"Input: {example[0]}\nOutput: {example[1]}"
                                          for example in few_shot_examples])
    input_example_string = f"Input: {input_example}"
    return f"<s>[INST] <<SYS>>\n{system_msg}\n{few_shot_examples_string}<</SYS>>\n{input_example_string}\nOutput: [/INST]"


def build_llama2_prompt_plain(few_shot_examples: Iterator[Tuple[str, str]], system_msg: str, input_example: str) -> str:
    # Plain text prompt, no special tokens
    # Works better for this task
    few_shot_examples_string = "\n".join([f"Input: {example[0]}\nOutput: {example[1]}"
                                          for example in few_shot_examples])
    input_example_string = f"Input: {input_example}"
    return f"{system_msg}\n{few_shot_examples_string}\n{input_example_string}\nOutput: "


def build_self_verification_prompt_plain(system_msg: str, input_example: str, candidate_entity: str, entity_class: str) -> str:
    # TODO: how to build the self-verification prompt properly with Few-NERD episodes?
    # How to choose which tokens to ask about in the few-shot examples?

    # For now, only build prompt for the input example (no few-shot examples)

    prompt_string = f'The input sentence: "{input_example}"\n'\
                    f'Is the word "{candidate_entity}" in the input sentence a {entity_class} entity? '\
                    f'Please answer with yes or no. Output: '
    return f"{system_msg}\n{prompt_string}"


def labels_from_output(llm_output: str, input_tokens: List[str], entity_class: str) -> List[str]:
    # TODO: What to do with different punctuation in input and output? e.g. ``` vs. ``
    # TODO: What to do when tokens don't match? e.g. "ohne filter" translated into "without filter"
    predicted_entities = extract_predicted_entities(llm_output)
    temp_input_tokens = [t for t in input_tokens]
    result_pairs = [[token, "O"] for token in input_tokens]
    result_index = 0

    for entity in predicted_entities:
        # NLTK tokenizer as described in Few-NERD paper
        entity_tokens = word_tokenize(entity)
        matched_tokens = 0
        while entity_tokens:
            if temp_input_tokens[0] == entity_tokens[0]:
                result_pairs[result_index][1] = entity_class
                matched_tokens += 1
                entity_tokens.pop(0)

            temp_input_tokens.pop(0)
            result_index += 1
    return [t[1] for t in result_pairs]
