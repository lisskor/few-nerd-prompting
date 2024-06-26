# FewNERD LLM Prompting

[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/lisskor/few-nerd-prompting/blob/main/LICENSE)

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

Download the [Few-NERD dataset from the link](https://ningding97.github.io/fewnerd/) 
(use sampled Few-NERD to get episode data; see the [Few-NERD repo](https://github.com/thunlp/Few-NERD) for detailed instructions).

The expected directory structure is:

```
Few-NERD
├── data
│   ├── episode-data
│   │   ├── inter
│   │   │   ├── dev_10_1.jsonl
│   │   │   ├── ...
│   │   └── intra
│   │       ├── dev_10_1.jsonl
│   │       ├── ...
│   ├── inter
│   │   ├── dev.txt
│   │   ├── test.txt
│   │   └── train.txt
│   ├── intra
│   │   ├── dev.txt
│   │   ├── test.txt
│   │   └── train.txt
│   └── supervised
│       ├── dev.txt
│       ├── test.txt
│       └── train.txt
├── ...
```

## 💪 Usage
To use LLMs for few-shot NER, run
```
python few_nerd_prompting/prompt_llm.py --pat CLARIFAI_PAT --data_file FEW_NERD_EPISODES_FILE --entity_classes CLASSES --output_file OUT_FILE
```

`--data_file` should be the path to a file containing Few-NERD episode data, e.g. `Few-NERD/data/episode-data/inter/dev_10_1.jsonl`. 
`--entity_classes` is the list of named entity types the model will be prompted to identify, e.g. `person location event`.

By default, the script will use the [Llama 2 7b-chat](https://clarifai.com/meta/Llama-2/models/llama2-7b-chat) model.
You can replace it with any other compatible model by changing the `--model_id`, `--user_id`, and `--app_id` parameters.

You may want to run the model on a subset of the full dataset, e.g.

```
python few_nerd_prompting/prompt_llm.py --first_episode 10 --n_episodes 5 --pat CLARIFAI_PAT --data_file FEW_NERD_EPISODES_FILE --entity_classes CLASSES --output_file OUT_FILE
```

will predict for episodes 10-14 of the dataset (processing 5 episodes starting with the 10th episode, one-based). 
If you processed the dataset in chunks like this, 
you can then merge the outputs into one file:

```
python few_nerd_prompting/join_sliced_outputs.py --input_files ALL_CHUNK_PREDICTIONS --output_file OUT_FILE
```

Finally, you can calculate the metrics to assess the quality of obtained predictions:

```
python few_nerd_prompting/evaluate_outputs.py --few_nerd_file GROUND_TRUTH_FILE --pred_file PREDICTIONS_FILE --entity_classes CLASSES
```

## 📚 Related Blogs
Read the related blog posts from Clarifai here:
- [Do LLMs Reign Supreme in Few-Shot NER?](https://www.clarifai.com/blog/do-llms-reign-supreme-in-few-shot-ner)
- [Do LLMs Reign Supreme in Few-Shot NER? Part II](https://www.clarifai.com/blog/do-llms-reign-supreme-in-few-shot-ner-part-ii)
- [Do LLMs Reign Supreme in Few-Shot NER? Part III]()


## Citation
If you found this repository useful, please consider citing:
```
@misc{korotkova-2024-llama-2-few-shot-ner,
  title={FewNERD LLM Prompting},
  author={Elizaveta Korotkova and Isaac Chung},
  year={2024},
  url={https://github.com/lisskor/few-nerd-prompting}
}
```