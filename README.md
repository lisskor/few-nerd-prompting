# FewNERD LLM Prompting

[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-370/) [![](https://img.shields.io/github/license/lisskor/few-nerd-prompting.svg)](https://github.com/lisskor/few-nerd-prompting/blob/main/LICENSE)

This repository uses [Clarifai](https://www.clarifai.com/) to prompt an LLM and perform few-shot NER on the [Few-NERD dataset](https://aclanthology.org/2021.acl-long.248/) using methods described in the [GPT-NER](https://github.com/ShuheWang1998/GPT-NER) paper. 

## 🚀 Installation

**Requirements**

In a new virtual environment, install the required packages using
```
pip install -r requirements.txt
```

**Clarifai**

To use Clarifai, [sign up for a Free Account](https://clarifai.com/signup) or login then [get a Personal Access Token](https://docs.clarifai.com/clarifai-basics/authentication/personal-access-tokens/) (PAT).

**Few-NERD**

Download the [Few-NERD from the link](https://ningding97.github.io/fewnerd/).

## 💪 Usage
To use LLMs for few-shot NER, run
```
python few_nerd_prompting/prompt_llm.py --pat CLARIFAI_PAT --data_file FEW_NERD_EPISODES_FILE --entity_classes CLASSES --output_file OUT_FILE
```

`--data_file` should be the path to a file containing Few-NERD episode data. 
`--entity_classes` is the list of named entity types the model will be prompted to identify, e.g. `person location event`.

By default, the script will use the [Llama 2 7b-chat](https://clarifai.com/meta/Llama-2/models/llama2-7b-chat) model.
You can replace it with any other compatible model by changing the `--model_id`, `--user_id`, and `--app_id` parameters.

You may want to run the model on a subset of the full dataset, e.g.

```
python few_nerd_prompting/prompt_llm.py --first_episode 0 --n_episodes 10 --pat CLARIFAI_PAT --data_file FEW_NERD_EPISODES_FILE --entity_classes CLASSES --output_file OUT_FILE
```

will only predict for the first 10 episodes of the dataset. If you processed the dataset in chunks like this, 
you can then merge the outputs into one file:

```
python few_nerd_prompting/join_sliced_outputs.py --input_files ALL_CHUNK_PREDICTIONS --output_file OUT_FILE
```

Finally, you can calculate the metrics to assess the quality of obtained predictions:

```
python few_nerd_prompting/evaluate_outputs.py --few_nerd_file GROUND_TRUTH_FILE --pred_file PREDICTIONS_FILE --entity_classes CLASSES
```
