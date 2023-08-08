# Duck News Reporters

This repository contains the source code for the programs written in this project. This README should guide the user through our file structure and a general hint on how to run the code.

Also see [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for a list of attributions. A plaintext version is included at [ATTRIBUTIONS.txt](ATTRIBUTIONS.txt) fo those without a markdown renderer.

## Directory structure

All code is contained within the `src` directory.

### Dataset

We included the dataset in `src/data/Horne2017_FakeNewsData/Buzzfeed`. Along with the raw dataset, it also contains intermediary cached files containing our contributions. These will be automatically loaded by our pipeline.

- `context.csv` - These are all the urls manually scraped for each input article. It also includes the summary we used.
- `context/` - This is a folder containing all the downloaded article content. This is not really intended for user browsing but it is cached here so we don't have to re-fetch them. They will be automatically joined when the dataset is loaded.
- `features_*.csv` - This contains all our features after running the pipeline. It acts as a cache so we don't have to run the whole pipeline multiple times.
- `tf_model/` - The saved model for our neural network classification model. This will be loaded by `main.py` when running our pipeline.

### Software

We include both helper code in `*.py` files and notebooks in `*.ipynb` files. Notebooks have been pre-run so they can be viewed by just opening them. The structure of python files is as follows:

- [classification.py](src/classification.py) - Contains the helpers for our classification models and a pipeline that will load all features.
- [dataset.py](src/dataset.py) - Contains helpers to load the dataset from the `data/`` folder
- [eda.py](src/eda.py) - Contains helper functions to perform data analysis.
- [non_latent_features.py](src/non_latent_features.py) - Contains helpers used to extract non latent features.
- [preprocess.py](src/preprocess.py) - Contains the helper to preprocess and tokenize input.
- [sentiment.py](src/sentiment.py) - Contains a legacy helper used to extract sentiment. This has been replaced by code within `non_latent_features.py`.
- [similarity.py](src/similarity.py) - Contains the summary extractor and similarity model.

---

- [main.py](src/main.py) - Takes the user through a run of our pipeline including getting any article, and prompting the user to enter 3 context articles based on a summary given to the user. Will build the features and perform an inference. **_Note: This is currently broken. It will run to the end but give funny results._**

---

- [EDA.ipynb](src/EDA.ipynb) - Contains all the data analysis done on the dataset and features.
- [pca_kmeans_analysis.ipynb](src/pca_kmeans_analysis.ipynb) - A notebook that investigates performing pca and kmeans on our features.
- [machine_learning.ipynb](src/machine_learning.ipynb) - Contains a notebook to run tests for our machine learning models.
- [deep_learning.ipynb](src/deep_learning.ipynb) - Contains a notebook to run our deep learning model.

## Running code

You will need to have `python>=3.10` installed and no guarantees are made about GPU support. The ducknewsreporters team uses a combination of Google Colab and local WSL2 systems with CUDA installed to run software. We believe that our code should automatically fal back to CPU.

To run the code:

```sh
# Make sure you have python>=3.10 installed
$ cd src
# This will install all the dependencies. (May take a while)
$ ./install
$ python3 main.py
```
