# MammoViT
This repository attempts to replicate the methods of Abdullah et al. (2023) in deploying a fine-tuning pipeline for the MammoViT custom vision transformer.

## Repository Structure

The repository is broken into two main components.

First, the top level notebook for running the INbreast MammoViT pipeline and its two supplement notebooks for pre-training MammoViT on ImageNet-1K and the second for implementing the original fine-tuning pipeline on the KAU-BCMD dataset.

The second component of this repository contains all crucial scripts for operation the noteboks which are contained within the `src`directory.


## Running the Pipeline

For operating the pipeline:
1. Install this repository within your google drive.
   1. I recommend opening a Google Colab instance, selecting the terminal, navigating to the directory level of your choosing, and `git clone` the repository.
1. Install the INbreast dataset at the same top level directory as this repository.
1. From Google Drive, select the option ellipses for INbreast_MammoViT_Pipeline.ipynb an select "open with Google Colab".
1. Once the notebook is active, again open the terminal, navigate to your current working directory, and run `pip install -r requirements.txt` .
1. With the environment established, run the cells of the notebook in order.
   1. Please note that some cells may require the session instance to be restarted. When necessary restart the session. Re-running these cells should not require the session to be restarted again.

## Finding Results


