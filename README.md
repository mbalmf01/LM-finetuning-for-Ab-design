# msc-project-source-code-files-22-23-mbalmf01
msc-project-source-code-files-22-23-mbalmf01 created by GitHub Classroom
LEVERAGING PROTEIN LANGUAGE MODELS FOR PREDICTION OF ANTIBODY PROPERTIES

## Abstract
Antibodies demand intricate and costly engineering to obtain molecules with properties that enable them to enter the clinic and be delivered as treatments for disease. These properties are typically considered only after years of extensive discovery and can result in significant delays to patients. Earlier consideration of these properties is warranted, however computational costs of structural modelling for derivation of these properties is a limiting factor. Transformers, and in particular protein language models, offer a solution to this problem by encoding protein structure and biophysical properties in the model latent embedding space. To probe the extent to which protein language models can be used to derive antibody biophysical properties, two distinct language models were compared for their ability to generate antibody sequence representations for a transfer learning task. Computationally expensive structural modelling was used to derive aggregation propensity scores for a set of diverse antibodies, and a series of neural networks were trained for target value prediction. We show that protein language models readily encode biophysical properties that can be used for training highly accurate supervised models. In addition, we show that these models can be deployed on large datasets for property prediction to flag problematic antibodies early in the discovery process.

## Use
This project uses a series of standalone Jupyter notebooks to guide you through the project code. All notebooks were implemented using Google Colab Pro. Each notebook will use one or several files that have been generated and stored in data_files to enable access to any notebook without requiring data generation steps in previous notebooks. Running the notebooks in series in fact will only work if the data is pulled from data_files, since a series of boring data merging/wrangling steps have been removed for ease of legibility of the code. Indeed, the purpose of this project is to demonstrate the power of language models for feature extraction and model training, and as such the focus is not on taking the user from A to B.

One key step in this project was the structural modelling and solubility prediction. The code for this has not been included in this repo as it was generated using code under patent, and the bureaucracy to get around this was too overbearing to warrant attempting.

This project showcases two language models, AbLang and Ankh. Feature extraction by sequence embedding is carried out using both LMs, and the resulting sequence representations are used to train either a classifier, for prediction of antibody germline lineage, or a regressor, for prediction of solubility (AKA aggregation propensity). In a final step, the language model ConvBERT is trained on Ankh sequence representations and solubility data to generate a model with exquisite model performance.

A rough working order is:
  1. extract_cluster.ipynb
  2. ablang_seq_embedding.ipynb
  3. ablang_embedding_to_vgene.ipynb
  4. ankh_seq_embedding.ipynb
  5. ankh_embedding to vgene.ipynb
  6. ablang_embedding_to_mapt_score.ipynb
  7. ankh_embedding_to_mapt_score.ipynb
  8. convert_to_mapt_score.ipynb
