import argparse
import csv
import datetime
import json
import gzip
import os
import numpy as np
import pandas as pd
import oyaml as yaml
import joblib

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam
import keras.backend as K
from keras.applications import vgg16
from keras.preprocessing.image import ImageDataGenerator

## HELPERS

## Importation des données

def load_embeddings(file_list, emb_dir):
    """
    Load saved embeddings from an embedding directory
    Parameters
    ----------
    file_list
    emb_dir
    Returns
    -------
    embeddings
    ignore_idxs
    """
    embeddings = []
    for idx, filename in enumerate(file_list):
        emb_path = os.path.join(emb_dir, os.path.splitext(filename)[0] + '.npy')
        embeddings.append(np.load(emb_path))

    return embeddings


def get_subset_split(annotation_data):
    """
    Get indices for train and validation subsets
    Parameters
    ----------
    annotation_data
    Returns
    -------
    train_idxs
    valid_idxs
    """

    # Get the audio filenames and the splits without duplicates
    data = annotation_data[['split', 'audio_filename']].drop_duplicates().sort_values('audio_filename')

    train_idxs = []
    valid_idxs = []
    for idx, (_, row) in enumerate(data.iterrows()):
        if row['split'] == 'train':
            train_idxs.append(idx)
        else:
            valid_idxs.append(idx)

    return np.array(train_idxs), np.array(valid_idxs)


def get_file_targets(annotation_data, labels):
    """
    Get file target annotation vector for the given set of labels
    Parameters
    ----------
    annotation_data
    labels
    Returns
    -------
    target_list
    """
    target_list = []
    file_list = annotation_data['audio_filename'].unique().tolist()

    for filename in file_list:
        file_df = annotation_data[annotation_data['audio_filename'] == filename]
        target = []

        for label in labels:
            count = 0

            for _, row in file_df.iterrows():
                if int(row['annotator_id']) == 0:
                    # If we have a validated annotation, just use that
                    count = row[label + '_presence']
                    break
                else:
                    count += row[label + '_presence']

            if count > 0:
                target.append(1.0)
            else:
                target.append(0.0)

        target_list.append(target)

    return np.array(target_list)

def prepare_data(train_file_idxs, test_file_idxs, embeddings,
                           target_list, standardize=True):
    """
    Prepare inputs and targets for training using training and evaluation indices.
    Parameters
    ----------
    train_file_idxs
    test_file_idxs
    embeddings
    target_list
    standardize
    Returns
    -------
    X_train
    y_train
    X_valid
    y_valid
    scaler
    """

    X_train = []
    y_train = []
    for idx in train_file_idxs:
        X_ = np.zeros((431,64,3))
        X_[:embeddings[idx].shape[0],:,:] = embeddings[idx]
        X_train.append(X_)
        y_train.append(target_list[idx])

    train_idxs = np.random.permutation(len(X_train))

    X_train = np.array(X_train)[train_idxs]
    X_train = np.swapaxes(X_train,1,2)
    X_train = np.flip(X_train,axis=1)
    mean=np.mean(X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],X_train.shape[3]),axis=0)
    std=np.std(X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2],X_train.shape[3]),axis=0)
    X_train[:,:,:,0]=(X_train[:,:,:,0]-mean[0])/std[0]
    X_train[:,:,:,1]=(X_train[:,:,:,1]-mean[1])/std[1]
    X_train[:,:,:,2]=(X_train[:,:,:,2]-mean[2])/std[2]
    y_train = np.array(y_train)[train_idxs]

    X_valid = []
    y_valid = []
    for idx in test_file_idxs:
        X_ = np.zeros((431,64,3))
        X_[:embeddings[idx].shape[0],:,:] = embeddings[idx]
        X_valid.append(X_)
        y_valid.append(target_list[idx])

    test_idxs = np.random.permutation(len(X_valid))

    X_valid = np.array(X_valid)[test_idxs]	
    X_valid = np.swapaxes(X_valid,1,2)
    X_valid = np.flip(X_valid,axis=1)   
    meanv=np.mean(X_valid.reshape(X_valid.shape[0]*X_valid.shape[1]*X_valid.shape[2],X_valid.shape[3]),axis=0)
    stdv=np.std(X_valid.reshape(X_valid.shape[0]*X_valid.shape[1]*X_valid.shape[2],X_valid.shape[3]),axis=0)
    X_valid[:,:,:,0]=(X_valid[:,:,:,0]-meanv[0])/stdv[0]
    X_valid[:,:,:,1]=(X_valid[:,:,:,1]-meanv[1])/stdv[1]
    X_valid[:,:,:,2]=(X_valid[:,:,:,2]-meanv[2])/stdv[2]
    y_valid = np.array(y_valid)[test_idxs]

    return X_train, y_train, X_valid, y_valid


## MODEL CONSTRUCTION

## Création du modèle

def createmodel(dropout_rate=0.3,l2_reg=0.1,ntime=431,nfreq=64,nchannel=3,num_classes=8):
    
    #input = Input(shape=(nfreq,ntime,nchannel))
    vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(nfreq,ntime,nchannel))
    output = vgg.layers[-1].output

    output = keras.layers.Flatten()(input)
    
    output = keras.layers.Dense(1024,kernel_regularizer=keras.regularizers.l2(l2_reg))(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Dropout(dropout_rate)(output)
    output = keras.layers.Dense(1024,kernel_regularizer=keras.regularizers.l2(l2_reg))(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Dropout(dropout_rate)(output)
    output = keras.layers.Dense(512,kernel_regularizer=keras.regularizers.l2(l2_reg))(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Dropout(dropout_rate)(output)
    output = keras.layers.Dense(512,kernel_regularizer=keras.regularizers.l2(l2_reg))(output)
    output = keras.layers.Activation('relu')(output)
    
    output = keras.layers.Dropout(dropout_rate)(output)
    output = keras.layers.Dense(num_classes, activation='sigmoid',kernel_regularizer=keras.regularizers.l2(l2_reg))(output)

    #model = Model(input, output)
    model = Model(vgg.input, output)
    
    return model

## GENERIC MODEL TRAINING

class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
    
        if y_val.shape[1]>8:
            y_val=np.delete(y_val,27,1)
            y_val=np.delete(y_val,22,1)
            y_val=np.delete(y_val,18,1)
            y_val=np.delete(y_val,13,1)
            y_val=np.delete(y_val,8,1)
            y_val=np.delete(y_val,3,1)
        
        y_pred = np.asarray(self.model.predict(X_val))

        res=[]
        for t in np.linspace(0,1,1000):
            yp=1*(y_pred>t)
            true_positives = np.sum(yp*y_val)
            predicted_positives = np.sum(yp)
            possible_positives = np.sum(y_val)
            p=true_positives / (predicted_positives + K.epsilon())
            r=true_positives / (possible_positives + K.epsilon())
            res.append([p,r])
        resa=np.array(res)
        resasort=resa[resa[:,0].argsort()]
        resasortu=[]
        for p in np.unique(resasort[:,0]):
            resasortu.append([p,np.max(resasort[resasort[:,0]==p][:,1])])
        resasortu[0]=[0,1]    
        resasortua=np.array(resasortu)
        s=0
        for i in range(len(resasortua)-1):
            dx=resasortua[i+1,0]-resasortua[i,0]
            fx=0.5*(resasortua[i+1,1]+resasortua[i,1])
            s=s+fx*dx      
        
        print('micro-AUPRC')
        print(round(s,4))
    
        if y_val.shape[1]>8:
            if s>0.65:
                os.makedirs('bestmodels/fine', exist_ok=True)
                self.model.save('bestmodels/fine/model'+str(round(s,4))+'.hdf5')
        else:
            if s>0.80:
                os.makedirs('bestmodels/coarse', exist_ok=True)
                self.model.save('bestmodels/coarse/model'+str(round(s,4))+'.hdf5')            
            
        
        return 

    def get_data(self):
        return self._data

class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)
            
        return X, y

def train_model(model, X_train, y_train, X_valid, y_valid, output_dir,
                loss=None, batch_size=16, num_epochs=100, patience=20,
                learning_rate=1e-5):
    """
    Train a model with the given data.

    Parameters
    ----------
    model
    X_train
    y_train
    output_dir
    batch_size
    num_epochs
    patience
    learning_rate

    Returns
    -------
    history

    """

    if loss is None:
        loss = 'binary_crossentropy'


    os.makedirs(output_dir, exist_ok=True)

    # Set up callbacks
    cb = []
    # checkpoint
    m=Metrics()
    cb.append(m)

    # monitor losses
    history_csv_file = os.path.join(output_dir, 'history.csv')
    cb.append(keras.callbacks.CSVLogger(history_csv_file, append=True,
                                        separator=','))

    model.summary()
    # Fit model
    model.compile(Adam(lr=learning_rate), loss=loss)

    datagen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1)
    training_generator = MixupGenerator(X_train, y_train, batch_size=batch_size, alpha=0.85, datagen=datagen)()
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size),steps_per_epoch=X_train.shape[0]//batch_size,
                    validation_data=(X_valid, y_valid),epochs=num_epochs, verbose=1,callbacks=cb)

    return history


## MODEL TRAINING

def train(annotation_path, taxonomy_path, emb_dir, output_dir, exp_id,
                    label_mode="fine", batch_size=64, num_epochs=100,
                    patience=20, learning_rate=1e-4, hidden_layer_size=128,
                    num_hidden_layers=0, l2_reg=1e-5, standardize=True,
                    timestamp=None):
    """
    Train and evaluate a model.

    Parameters
    ----------
    dataset_dir
    emb_dir
    output_dir
    exp_id
    label_mode
    batch_size
    test_ratio
    num_epochs
    patience
    learning_rate
    hidden_layer_size
    l2_reg
    standardize
    timestamp

    Returns
    -------

    """

    # Load annotations and taxonomy
    print("* Loading annotations...")
    annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.load(f, Loader=yaml.Loader)

    file_list = annotation_data['audio_filename'].unique().tolist()
    
    full_fine_target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                               for coarse_id, fine_dict in taxonomy['fine'].items()
                               for fine_id, fine_label in fine_dict.items()]
    fine_target_labels = [x for x in full_fine_target_labels
                            if x.split('_')[0].split('-')[1] != 'X']
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]

    # For fine, we include incomplete labels in targets for computing the loss
    fine_target_list = get_file_targets(annotation_data, full_fine_target_labels)
    coarse_target_list = get_file_targets(annotation_data, coarse_target_labels)
    train_file_idxs, test_file_idxs = get_subset_split(annotation_data)

    if label_mode == "fine":
        target_list = fine_target_list
        labels = fine_target_labels
    elif label_mode == "coarse":
        target_list = coarse_target_list
        labels = coarse_target_labels
    else:
        raise ValueError("Invalid label mode: {}".format(label_mode))


    print('* Loading mel-spectrograms...')
    embeddings = load_embeddings(file_list, emb_dir)

    print('* Preparing data...')
    X_train, y_train, X_valid, y_valid = prepare_data(train_file_idxs, test_file_idxs, embeddings,target_list, standardize=standardize)
    nsamples, nfreq, ntime, nchannel = X_train.shape
    num_classes = len(labels)

    print('* Creating the model...')
    dropout_rate=0.3
    model = createmodel(dropout_rate,l2_reg,ntime,nfreq,nchannel,num_classes)

    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    results_dir = os.path.join(output_dir, exp_id, timestamp)

    if label_mode == "fine":
        full_coarse_to_fine_terminal_idxs = np.cumsum(
            [len(fine_dict) for fine_dict in taxonomy['fine'].values()])
        incomplete_fine_subidxs = [len(fine_dict) - 1 if 'X' in fine_dict else None
                                   for fine_dict in taxonomy['fine'].values()]
        coarse_to_fine_end_idxs = np.cumsum([len(fine_dict) - 1 if 'X' in fine_dict else len(fine_dict)
                                             for fine_dict in taxonomy['fine'].values()])

        # Create loss function that only adds loss for fine labels for which
        # the we don't have any incomplete labels
        def masked_loss(y_true, y_pred):
            loss = None
            for coarse_idx in range(len(full_coarse_to_fine_terminal_idxs)):
                true_terminal_idx = full_coarse_to_fine_terminal_idxs[coarse_idx]
                true_incomplete_subidx = incomplete_fine_subidxs[coarse_idx]
                pred_end_idx = coarse_to_fine_end_idxs[coarse_idx]

                if coarse_idx != 0:
                    true_start_idx = full_coarse_to_fine_terminal_idxs[coarse_idx-1]
                    pred_start_idx = coarse_to_fine_end_idxs[coarse_idx-1]
                else:
                    true_start_idx = 0
                    pred_start_idx = 0

                if true_incomplete_subidx is None:
                    true_end_idx = true_terminal_idx

                    sub_true = y_true[:, true_start_idx:true_end_idx]
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx]

                else:
                    # Don't include incomplete label
                    true_end_idx = true_terminal_idx - 1
                    true_incomplete_idx = true_incomplete_subidx + true_start_idx
                    assert true_end_idx - true_start_idx == pred_end_idx - pred_start_idx
                    assert true_incomplete_idx == true_end_idx

                    # 1 if not incomplete, 0 if incomplete
                    mask = K.expand_dims(1 - y_true[:, true_incomplete_idx])

                    # Mask the target and predictions. If the mask is 0,
                    # all entries will be 0 and the BCE will be 0.
                    # This has the effect of masking the BCE for each fine
                    # label within a coarse label if an incomplete label exists
                    sub_true = y_true[:, true_start_idx:true_end_idx] * mask
                    sub_pred = y_pred[:, pred_start_idx:pred_end_idx] * mask

                if loss is not None:
                    loss += K.sum(K.binary_crossentropy(sub_true, sub_pred))
                else:
                    loss = K.sum(K.binary_crossentropy(sub_true, sub_pred))

            return loss
        loss_func = masked_loss
    else:
        loss_func = None

    print("* Training model...")
    history = train_model(model, X_train, y_train, X_valid, y_valid,
                          results_dir, loss=loss_func, batch_size=batch_size,
                          num_epochs=num_epochs, patience=patience,
                          learning_rate=learning_rate)

    print("* Computing model predictions...")
    results = {}
    results['train'] = predict(embeddings, train_file_idxs, model)
    results['test'] = predict(embeddings, test_file_idxs, model)
    results['train_history'] = history.history

    print('* Saving model predictions...')
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    for aggregation_type, y_pred in results['test'].items():
        generate_output_file(y_pred, test_file_idxs, results_dir, file_list,
                             aggregation_type, label_mode, taxonomy)


## MODEL EVALUATION

def predict(embeddings, test_file_idxs, model, scaler=None):
    """
    Evaluate the output of a classification model.

    Parameters
    ----------
    embeddings
    test_file_idxs
    model
    scaler

    Returns
    -------
    results
    """
    y_pred_max = []
    y_pred_mean = []
    y_pred_softmax = []

    X_valid = []
    for idx in test_file_idxs:
        X_ = np.zeros((431,64,3))
        X_[:embeddings[idx].shape[0],:,:] = embeddings[idx]
        X_valid.append(X_)

    X_valid = np.array(X_valid)
    X_valid = np.swapaxes(X_valid,1,2)
    X_valid = np.flip(X_valid,axis=1)   
    meanv=np.mean(X_valid.reshape(X_valid.shape[0]*X_valid.shape[1]*X_valid.shape[2],X_valid.shape[3]),axis=0)
    stdv=np.std(X_valid.reshape(X_valid.shape[0]*X_valid.shape[1]*X_valid.shape[2],X_valid.shape[3]),axis=0)
    X_valid[:,:,:,0]=(X_valid[:,:,:,0]-meanv[0])/stdv[0]
    X_valid[:,:,:,1]=(X_valid[:,:,:,1]-meanv[1])/stdv[1]
    X_valid[:,:,:,2]=(X_valid[:,:,:,2]-meanv[2])/stdv[2]

    pred = model.predict(X_valid).tolist()

    results = {
        'max': pred,
        'mean': pred,
        'softmax': pred
    }

    return results


def generate_output_file(y_pred, test_file_idxs, results_dir, file_list,
                         aggregation_type, label_mode, taxonomy):
    """
    Write the output file containing model predictions

    Parameters
    ----------
    y_pred
    test_file_idxs
    results_dir
    file_list
    aggregation_type
    label_mode
    taxonomy

    Returns
    -------

    """
    output_path = os.path.join(results_dir, "output_{}.csv".format(aggregation_type))
    test_file_list = [file_list[idx] for idx in test_file_idxs]

    coarse_fine_labels = [["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                             for fine_id, fine_label in fine_dict.items()]
                           for coarse_id, fine_dict in taxonomy['fine'].items()]

    full_fine_target_labels = [fine_label for fine_list in coarse_fine_labels
                                          for fine_label in fine_list]
    coarse_target_labels = ["_".join([str(k), v])
                            for k,v in taxonomy['coarse'].items()]


    with open(output_path, 'w') as f:
        csvwriter = csv.writer(f)

        # Write fields
        fields = ["audio_filename"] + full_fine_target_labels + coarse_target_labels
        csvwriter.writerow(fields)

        # Write results for each file to CSV
        for filename, y, in zip(test_file_list, y_pred):
            row = [filename]

            if label_mode == "fine":
                fine_values = []
                coarse_values = [0 for _ in range(len(coarse_target_labels))]
                coarse_idx = 0
                fine_idx = 0
                for coarse_label, fine_label_list in zip(coarse_target_labels,
                                                         coarse_fine_labels):
                    for fine_label in fine_label_list:
                        if 'X' in fine_label.split('_')[0].split('-')[1]:
                            # Put a 0 for other, since the baseline doesn't
                            # account for it
                            fine_values.append(0.0)
                            continue

                        # Append the next fine prediction
                        fine_values.append(y[fine_idx])

                        # Add coarse level labels corresponding to fine level
                        # predictions. Obtain by taking the maximum from the
                        # fine level labels
                        coarse_values[coarse_idx] = max(coarse_values[coarse_idx],
                                                        y[fine_idx])
                        fine_idx += 1
                    coarse_idx += 1

                row += fine_values + coarse_values

            else:
                # Add placeholder values for fine level
                row += [0.0 for _ in range(len(full_fine_target_labels))]
                # Add coarse level labels
                row += list(y)

            csvwriter.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("taxonomy_path")
    parser.add_argument("emb_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("exp_id", type=str)    

    parser.add_argument("--hidden_layer_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--l2_reg", type=float, default=1e-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no_standardize", action='store_true')
    parser.add_argument("--label_mode", type=str, choices=["fine", "coarse"],
                        default='coarse')

    args = parser.parse_args()

    # save args to disk
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(args.output_dir, args.exp_id, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    kwarg_file = os.path.join(out_dir, "hyper_params.json")
    with open(kwarg_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    train(args.annotation_path,
                    args.taxonomy_path,
                    args.emb_dir,
                    args.output_dir,
                    args.exp_id,
                    label_mode=args.label_mode,
                    batch_size=args.batch_size,
                    num_epochs=args.num_epochs,
                    patience=args.patience,
                    learning_rate=args.learning_rate,
                    hidden_layer_size=args.hidden_layer_size,
                    num_hidden_layers=args.num_hidden_layers,
                    l2_reg=args.l2_reg,
                    standardize=(not args.no_standardize),
                    timestamp=timestamp)
