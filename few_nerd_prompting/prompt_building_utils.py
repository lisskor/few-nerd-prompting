from typing import List, Tuple, Iterator

from nltk import word_tokenize

TAG_START = "@@"
TAG_END = "##"


def make_output_example(sentence: List[str], labels: List[str], entity_class: str) -> str:
    """
    Add entity start and end tags to the input text to obtain an output example (for a single entity class at a time).
    E.g. sentence = ["I", "am", "in", "Tallinn", "."], 
         labels = ["O", "O", "O", "location", "O"],
         entity_class = "location"
    -> 'I am in @@Tallinn## .'
    
    :param sentence: list of tokens
    :param labels: list of token labels
    :param entity_class: entity class (e.g. "person", "location" etc.)
    :return: output text with tags around the entities belonging to the class in question
    """
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
    """
    Check whether the sentence with entity tags is well-formed: 
    each TAG_START has a corresponding TAG_END,
    the entities do not overlap (each one is closed before next is opened).
    E.g. "We live in @@New York##." -> True
         "We live in @@New York." -> False
         "We live in @@New @@York##." -> False
    
    :param sentence: string which may contain entity tags
    :return: True if sentence is well-formed, False otherwise
    """
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
    """
    Extract predicted entities from sentences with entity start and end tags.
    E.g. "I am in @@New York##." -> ['New York']
    
    :param sentence: sentence string which may include named entities marked with start and end tags
    :return: list of marked entities
    """
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


def build_llama2_prompt(
        few_shot_examples: Iterator[Tuple[str, str]], system_msg: str, instr_msg: str, input_example: str
) -> str:
    """
    Create prompt for Llama 2 with <s>[INST] <<SYS>> ... <</SYS>> ... [/INST]
    <s> - the beginning of the entire sequence
    <<SYS>> - the beginning of the system message
    <</SYS>> - the end of the system message
    [INST] - the beginning of some instructions
    [/INST] - the end of the instructions
    
    :param few_shot_examples: iterable containing pairs of input and output few-shot examples
    :param system_msg: system message
    :param instr_msg: instruction message
    :param input_example: input example to predict for
    :return: full prompt
    """
    few_shot_examples_string = "\n".join([f"Input: {example[0]}\nOutput: {example[1]}"
                                          for example in few_shot_examples])
    input_example_string = f"Input: {input_example}"
    return f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n{instr_msg}\n{few_shot_examples_string}\n{input_example_string}\nOutput: [/INST]"


def build_llama2_prompt_plain(
        few_shot_examples: Iterator[Tuple[str, str]], system_msg: str, instr_msg: str, input_example: str
) -> str:
    """
    Create plain text prompt for Llama 2 without special tokens.
    
    :param few_shot_examples: iterable containing pairs of input and output few-shot examples
    :param system_msg: system message
    :param instr_msg: instruction message
    :param input_example: input example to predict for
    :return: full prompt
    """
    few_shot_examples_string = "\n".join([f"Input: {example[0]}\nOutput: {example[1]}"
                                          for example in few_shot_examples])
    input_example_string = f"Input: {input_example}"
    return f"{system_msg} {instr_msg}\n{few_shot_examples_string}\n{input_example_string}\nOutput: "


def build_self_verification_prompt_plain(system_msg: str, input_example: str, candidate_entity: str, entity_class: str) -> str:
    """
    Create plain text prompt for the self-verification step.
    
    :param system_msg: system message
    :param input_example: input sentence
    :param candidate_entity: predicted entity
    :param entity_class: the entity class to verify
    :return: full prompt
    """
    # TODO: how to build the self-verification prompt properly with Few-NERD episodes?
    # How to choose which tokens to ask about in the few-shot examples?

    # For now, only build prompt for the input example (no few-shot examples)

    prompt_string = f'The input sentence: "{input_example}"\n'\
                    f'Is the word "{candidate_entity}" in the input sentence a {entity_class} entity? '\
                    f'Please answer with yes or no. Output: '
    return f"{system_msg}\n{prompt_string}"


def labels_from_output(llm_output: str, input_tokens: List[str], entity_class: str) -> List[str]:
    """
    Given a generated output sentence, a list of original input tokens for that sentence, and an entity class,
    create a list of predicted token labels.
    E.g. llm_output = " the @@geisel library## is considered his legacy at ucsd.",
         tokens = ["the", "geisel", "library", "is", "considered", "his", "legacy", "at", "ucsd", "."],
         entity_class = "building"
    -> ["O", "building", "building", "O", "O", "O", "O", "O", "O", "O"]
    
    :param llm_output: a generated sentence with predicted entities marked with start and end tags
    :param input_tokens: list of original input tokens
    :param entity_class: entity class
    :return: list of token labels
    """
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
