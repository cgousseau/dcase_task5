# dcase_task5

This repository contains my submission for the Task 5 (Urban Sound Tagging) of the the DCASE 2019 Challenge.
The code from the baseline (https://github.com/sonyc-project/urban-sound-tagging-baseline) was used as a starting point.

It is part of my master thesis internship done at Orange Labs Lannion

## Dowloading the data
You can download the data at: https://zenodo.org/record/3233082#.XPpfYCBS-Uk

## Extracting the input features
The feature engineering uses the method presented in the book \textit{Hands-On Transfer Learning with Python}. First the recordings are re-sampled using a sampling rate of 22050Hz. Then three features are extracted from these signals:
- the mel-spectrograms using 64 mel-bands and a hop length of 512 thus resulting a 64 rows x 431 colums image
- the averaged value of the harmonic and percussive components (64 rows x 431 colums image)
- the derivative of the log-mel spectrograms (64 rows x 431 colums image)

``` python extract_embedding.py data/annotations.csv data features  ```

This transforms the sounds in the repository 'data' using the file 'annotations.csv' and store them in the repository 'features'

## Training a model
The model is a VGG-16 network pre-trained on the ImageNet dataset. For more details, see the technical report.

``` python classify.py data/annotations.csv dcase-ust-taxonomy.yaml features/mel output baseline_coarse ```

## Geneating output files
Then an output file following the Dcase challenge format is generated:

``` python generate_predictions.py data/annotations.csv  dcase-ust-taxonomy.yaml features/mel  output baseline_coarse ```

## Evaluating the performance
The metrics of the challenge are computed:

```  python evaluate_predictions.py output/baseline_coarse/*/output_mean.csv data/annotations.csv dcase-ust-taxonomy.yaml ```

'*' is the timestamp
