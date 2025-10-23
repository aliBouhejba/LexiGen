# LexiGen: GDPR Compliance Corpus

LexiGen is a research corpus designed to support the analysis and evaluation of privacy policies for GDPR compliance on article-level compliance tasks.  

## Folder Structure

### `benchmark/`
Contains training and evaluation datasets, as well as scripts for model fine-tuning and evaluation.

- `data/` – Training and evaluation data in JSON format.  
- `scripts/` – Fine-tuning and evaluation scripts for different model families:
  - `decoders-only/` – Scripts for decoder-based models such as Mistral-7B-V0.1, Cimphony, and GPT-OSS-20.  
  - `encoders-only/` – Scripts for encoder-based models such as BERT and LegalBERT.

### `corpus_creation/`
Contains the resources and scripts used to build and structure the LexiGen corpus.

- `data/` – Obligations file, structured GDPR articles, and the main synthetic corpus.  
- `scripts/` – Scripts for extracting obligations, scraping GDPR texts, and creating synthetic corpus.

### `requirements.txt`
Lists the Python dependencies needed to run the project.



### `README.md`
Project description and folder structure overview.
