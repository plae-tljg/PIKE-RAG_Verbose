# Biology Q&A

## Text Book

The used biology book is hosted in GitHub [biology-book](https://github.com/philschatz/biology-book).

The Table of Contents is in [./SUMMARY.md](./SUMMARY.md)

## Test Dataset

The used test dataset is [MMLU/college_biology](https://huggingface.co/datasets/cais/mmlu).

## Pipeline and Script Description

Assume that you are in the directory `examples/`:

```sh
# Using the general chunking script to process original documents.
python chunking.py biology/configs/chunking.yml

# Using the general qa script to do Q&A testing on the given testing suite.
python qa.py biology/configs/qa.yml
```
