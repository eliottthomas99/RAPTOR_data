# RAPTOR Dataset and Scripts Repository

This repository contains datasets and scripts used in the RAPTOR (Robust Automated Parsing for Tabular Object Recognition) research project. The datasets included here are adaptations of ICDAR 2013 and DOCILE, modified to suit the research context as described in the supplementary materials and the paper.

## Repository Structure

- data/
    - ICDAR-2013/
        - ground_truth/ # Ground truth annotations for ICDAR 2013
        - images_and_ocr/ # Contains PNG images and DOCTR OCR files for ICDAR 2013
    - DOCILE/
        - ground_truth/ # Ground truth annotations for DOCILE
        - images_and_ocr/ # Contains PNG images and DOCTR OCR files for DOCILE
- scripts/
    - docile_get_subset.py # Python script to extract subsets from DOCILE dataset
    - icdar13_get_subset.py # Python script to extract subsets from ICDAR 2013 dataset
- headers_dict.json # JSON file containing the dictionary of headers used in the research

## Dataset Information

The datasets (`ICDAR-2013` and `DOCILE`) in the `data/` directory have been modified for the specific requirements of the RAPTOR research project. Each dataset includes:
- **ground_truth/**: Contains annotated ground truth data necessary for training and evaluation.
- **images_and_ocr/**: Contains PNG images and OCR text files (DOCTR) corresponding to each dataset.

## Scripts

### Data Subset Extraction
- **docile_get_subset.py**: Python script used to extract subsets from the DOCILE dataset.
- **icdar13_get_subset.py**: Python script used to extract subsets from the ICDAR-2013 dataset.

These scripts facilitate the extraction of specific subsets from the original datasets, tailored for experimental purposes outlined in the research.

## Header Dictionary

The `headers_dict.json` file provides a dictionary of headers used in the RAPTOR research. 
