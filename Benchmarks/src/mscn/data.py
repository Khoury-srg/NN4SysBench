import os
import csv

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import dataset
from .util import *


def encode_query(queries, num_materialized_samples, save_path, test_data_name):
    joins = []
    predicates = []
    tables = []
    samples = []
    label = []
    data_raw = list(list(rec.split('#')) for rec in queries)
    table2vec, column2vec, op2vec, join2vec, column_min_max_vals, min_val, max_val, max_num_samples, max_num_joins, max_num_predicates = torch.load(
        os.path.join(save_path, 'saved_dicts.pt'))
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)
    for row in data_raw:
        tables.append(row[0].split(','))
        joins.append(row[1].split(','))
        predicates.append(row[2].split(','))
        if int(row[3]) < 1:
            print("Queries must have non-zero cardinalities")
            exit(1)
        label.append(row[3])
    # print("Loaded queries")
    if num_materialized_samples > 0:
        # Load bitmaps
        num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
        with open('workloads/'+test_data_name + ".bitmaps", 'rb') as f:
            for i in range(len(tables)):
                four_bytes = f.read(4)
                if not four_bytes:
                    print("Error while reading 'four_bytes'")
                    exit(1)
                num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
                bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
                for j in range(num_bitmaps_curr_query):
                    # Read bitmap
                    bitmap_bytes = f.read(num_bytes_per_bitmap)
                    if not bitmap_bytes:
                        print("Error while reading 'bitmap_bytes'")
                        exit(1)
                    bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
                samples.append(bitmaps)
        print("Loaded bitmaps")
    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]
    # Get feature encoding and proper normalization
    samples_test = encode_samples(tables, samples, table2vec)
    predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    # print("Number of test samples: {}".format(len(labels_test)))
    # max_num_samples = max(len(i) for i in samples_test)
    # max_num_predicates = max([len(p) for p in predicates_test])
    # max_num_joins = max([len(j) for j in joins_test])

    dataset, indexes = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_samples, max_num_joins, max_num_predicates, sample_feats, predicate_feats, join_feats)

    return dataset, min_val, max_val, max_num_samples, max_num_predicates, max_num_joins, indexes


def load_data(file_name, num_materialized_samples):
    joins = []
    predicates = []
    tables = []
    samples = []
    label = []

    # Load queries
    with open(file_name + ".csv", 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            label.append(row[3])
    # print("Loaded queries")
    if num_materialized_samples != 0:
        # Load bitmaps
        num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
        with open(file_name + ".bitmaps", 'rb') as f:
            for i in range(len(tables)):
                four_bytes = f.read(4)
                if not four_bytes:
                    print("Error while reading 'four_bytes'")
                    exit(1)
                num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
                bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
                for j in range(num_bitmaps_curr_query):
                    # Read bitmap
                    bitmap_bytes = f.read(num_bytes_per_bitmap)
                    if not bitmap_bytes:
                        print("Error while reading 'bitmap_bytes'")
                        exit(1)
                    bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
                samples.append(bitmaps)
        print("Loaded bitmaps")

    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]

    return joins, predicates, tables, samples, label


def load_and_encode_train_data(num_queries, num_materialized_samples):
    file_name_queries = "data/train"
    file_name_column_min_max_vals = "data/column_min_max_vals.csv"

    joins, predicates, tables, samples, label = load_data(file_name_queries, num_materialized_samples)

    # Get column name dict
    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)

    # Get table name dict
    table_names = get_all_table_names(tables)
    table2vec, idx2table = get_set_encoding(table_names)

    # Get operator name dict
    operators = get_all_operators(predicates)
    op2vec, idx2op = get_set_encoding(operators)

    # Get join name dict
    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    # Get min and max values for each column
    with open(file_name_column_min_max_vals, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    # Get feature encoding and proper normalization
    samples_enc = encode_samples(tables, samples, table2vec)
    predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    label_norm, min_val, max_val = normalize_labels(label)

    # Split in training and validation samples
    num_train = int(num_queries * 0.9)
    num_test = num_queries - num_train

    samples_train = samples_enc[:num_train]
    predicates_train = predicates_enc[:num_train]
    joins_train = joins_enc[:num_train]
    labels_train = label_norm[:num_train]

    samples_test = samples_enc[num_train:num_train + num_test]
    predicates_test = predicates_enc[num_train:num_train + num_test]
    joins_test = joins_enc[num_train:num_train + num_test]
    labels_test = label_norm[num_train:num_train + num_test]

    # print("Number of training samples: {}".format(len(labels_train)))
    # print("Number of validation samples: {}".format(len(labels_test)))

    max_num_samples = max(len(i) for i in samples_enc)
    max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))

    dicts = [table2vec, column2vec, op2vec, join2vec]
    train_data = [samples_train, predicates_train, joins_train]
    test_data = [samples_test, predicates_test, joins_test]
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_samples, max_num_joins, max_num_predicates, train_data, test_data


def zero_padding(samples, predicates, joins, labels, max_num_samples, max_num_joins, max_num_predicates, sample_feats, predicate_feats, join_feats):
    """Add zero-padding and wrap as tensor dataset."""
    ## Padding for NN verification input
    max_dim = max(sample_feats, predicate_feats, join_feats)
    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor = np.vstack(sample)
        num_pad = max_num_joins + 1 - sample_tensor.shape[0]
        sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
        sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
        sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    # sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    # sample_masks = torch.FloatTensor(sample_masks)
    sample_aggr = np.pad(np.concatenate([sample_tensors, sample_masks], axis=-1), pad_width=[(0,0),(0,0),(0,max_dim-sample_feats)])


    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    # predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    # predicate_masks = torch.FloatTensor(predicate_masks)
    predict_aggr = np.pad(np.concatenate([predicate_tensors, predicate_masks], axis=-1),
                         pad_width=[(0, 0), (0, 0), (0, max_dim - predicate_feats)])

    join_masks = []
    join_tensors = []
    for join in joins:
        join_tensor = np.vstack(join)
        num_pad = max_num_joins - join_tensor.shape[0]
        join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
        join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
        join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
        join_tensors.append(np.expand_dims(join_tensor, 0))
        join_masks.append(np.expand_dims(join_mask, 0))
    join_tensors = np.vstack(join_tensors)
    # join_tensors = torch.FloatTensor(join_tensors)
    join_masks = np.vstack(join_masks)
    # join_masks = torch.FloatTensor(join_masks)
    join_aggr = np.pad(np.concatenate([join_tensors, join_masks], axis=-1),
                          pad_width=[(0, 0), (0, 0), (0, max_dim - join_feats)])
    input_tensor = np.concatenate([sample_aggr, predict_aggr, join_aggr], axis=1)
    input_tensor = torch.FloatTensor(input_tensor)

    target_tensor = torch.FloatTensor(labels)
    # return sample_tensors, predicate_tensors, join_tensors, target_tensor, sample_masks, predicate_masks, join_masks
    return input_tensor, target_tensor


def make_dataset(samples, predicates, joins, labels, max_num_samples, max_num_joins, max_num_predicates, sample_feats, predicate_feats, join_feats):
    new_samples = []
    new_predicates = []
    new_joins = []
    new_labels = []
    indexes = []
    for index, (i,j,k,v) in enumerate(zip(samples, predicates, joins, labels)):
        if len(i) > max_num_samples or len(j) > max_num_predicates or len(k) > max_num_joins:
            continue
        else:
            new_samples.append(i)
            new_predicates.append(j)
            new_joins.append(k)
            new_labels.append(v)
            indexes.append(index)
    # print(f"Number of test samples: {len(new_samples)}")
    return dataset.TensorDataset(*zero_padding(new_samples, new_predicates, new_joins, new_labels, max_num_samples, max_num_joins, max_num_predicates, sample_feats, predicate_feats, join_feats)), indexes
    # return dataset.TensorDataset(sample_tensors, predicate_tensors, join_tensors, target_tensor, sample_masks,
    #                              predicate_masks, join_masks)


def get_train_datasets(num_queries, num_materialized_samples):
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_samples, max_num_joins, max_num_predicates, train_data, test_data = load_and_encode_train_data(
        num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts
    # Train model
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)
    train_dataset, indexes = make_dataset(*train_data, labels=labels_train, max_num_samples=max_num_samples, max_num_joins=max_num_joins,
                                 max_num_predicates=max_num_predicates, sample_feats=sample_feats, predicate_feats=predicate_feats, join_feats=join_feats)
    print("Created TensorDataset for training data")
    test_dataset, indexes = make_dataset(*test_data, labels=labels_test, max_num_samples=max_num_samples, max_num_joins=max_num_joins,
                                max_num_predicates=max_num_predicates, sample_feats=sample_feats, predicate_feats=predicate_feats, join_feats=join_feats)
    print("Created TensorDataset for validation data")
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_samples, max_num_joins, max_num_predicates, train_dataset, test_dataset
