# dcase_task5

This repository contains my submission for the Task 5 (Urban Sound Tagging) of the the DCASE 2019 Challenge.
The code from the baseline (https://github.com/sonyc-project/urban-sound-tagging-baseline) was used as a starting point.

It is part of my master thesis internship done at Orange Labs Lannion

## Setting up an environment
You will need the environment presented in the baseline (https://github.com/sonyc-project/urban-sound-tagging-baseline) + two additional libraries: soundfile and librosa (using 'pip install <name_of_the_library>'). Using tensorflow-gpu is recommended.

## Dowloading the data
You can download the data at: https://zenodo.org/record/3233082#.XPpfYCBS-Uk and store the folders 'train' and 'validate' into the folder 'data'

## Extracting the input features
The feature engineering uses the method presented in the book 'Hands-On Transfer Learning with Python'. First the recordings are re-sampled using a sampling rate of 22050Hz. Then three features are extracted from these signals:
- the mel-spectrograms using 64 mel-bands and a hop length of 512 thus resulting a 64 rows x 431 colums image
- the averaged value of the harmonic and percussive components (64 rows x 431 colums image)
- the derivative of the log-mel spectrograms (64 rows x 431 colums image)

``` python extract_mel.py data/annotations.csv data features  ```

This transforms the sounds in the repository 'data' using the file 'annotations.csv' and store them in the repository 'features'

## Training a model
The model is a VGG-16 network pre-trained on the ImageNet dataset. For more details, see the technical report.

``` python classify.py data/annotations.csv dcase-ust-taxonomy.yaml features/mel output baseline_coarse ```

This trains a model using the input features in 'features/mel'

## Generating output files
Then an output file following the Dcase challenge format is generated:

``` python generate_predictions.py data/annotations.csv  dcase-ust-taxonomy.yaml features/mel  output baseline_coarse ```

## Evaluating the performance
The metrics of the challenge are computed:

```  python evaluate_predictions.py output/baseline_coarse/*/output_mean.csv data/annotations.csv dcase-ust-taxonomy.yaml ```

'*' is the timestamp

## Results:

### Coarse level model

#### Coarse level evaluation:
======================
 * Micro AUPRC:           0.8261913279303387
 * Micro F1-score (@0.5): 0.743362831858407
 * Macro AUPRC:           0.611261794893059
 * Coarse Tag AUPRC:
      - 1: 0.8684264612721558
      - 2: 0.6049144982578505
      - 3: 0.5650667475752684
      - 4: 0.6889917351156993
      - 5: 0.9205722885890331
      - 6: 0.1799830711437796
      - 7: 0.9479529102624412
      - 8: 0.11418664692824343

### Fine level model

#### Fine level evaluation:
======================
 * Micro AUPRC:           0.7014738063991779
 * Micro F1-score (@0.5): 0.6127583108715184
 * Macro AUPRC:           0.4724765604745462
 * Coarse Tag AUPRC:
      - 1: 0.6531594643746426
      - 2: 0.2512843567121016
      - 3: 0.5434017171930277
      - 4: 0.31237853424904555
      - 5: 0.8335765730854546
      - 6: 0.11773996273721583
      - 7: 0.8865961250349249
      - 8: 0.18167575040995732

#### Coarse level evaluation:
======================
 * Micro AUPRC:           0.7740104836386648
 * Micro F1-score (@0.5): 0.6375779162956368
 * Macro AUPRC:           0.567213952487685
 * Coarse Tag AUPRC:
      - 1: 0.8523622398735744
      - 2: 0.2701405020236839
      - 3: 0.5434017171930277
      - 4: 0.6421566170571933
      - 5: 0.9037666467289218
      - 6: 0.18743184569126736
      - 7: 0.9567763009238545
      - 8: 0.18167575040995732
