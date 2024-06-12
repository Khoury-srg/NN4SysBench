import argparse
import time
import os
import pytorch_warmup as warmup
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mscn.util import *
from mscn.data import get_train_datasets, load_data, make_dataset
from mscn.model import SetConv


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        inputs, samples = data_batch

        if cuda:
        #     samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
        #     sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        # samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
        #     targets)
        # sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
        #     join_masks)
            inputs = inputs.cuda()
            inputs = Variable(inputs)

        t = time.time()
        outputs = model(inputs)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))
    return qerror


def evaluate(workload_name, batch_size, save_path, hid, cuda=False, materialize=False):
    if materialize:
        num_materialized_samples = 1000
    else:
        num_materialized_samples = 0
    table2vec, column2vec, op2vec, join2vec, column_min_max_vals, min_val, max_val, max_num_samples, max_num_joins, max_num_predicates = torch.load(
        os.path.join(save_path, 'saved_dicts.pt'))
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    # Load test data
    file_name = "workloads/" + workload_name
    joins, predicates, tables, samples, label = load_data(file_name, num_materialized_samples)

    # Get feature encoding and proper normalization
    samples_test = encode_samples(tables, samples, table2vec)
    predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    print("Number of test samples: {}".format(len(labels_test)))

    # max_num_samples = max(len(i) for i in samples_test)
    # max_num_predicates = max([len(p) for p in predicates_test])
    # max_num_joins = max([len(j) for j in joins_test])

    model = SetConv(sample_feats, predicate_feats, join_feats, hid, max_num_samples, max_num_joins, max_num_predicates)
    model.load_state_dict(torch.load(os.path.join(save_path, 'model-dict.pt'), map_location='cuda' if cuda else 'cpu'))
    model.eval()
    # Get test set predictions
    test_data = make_dataset(samples_test, predicates_test, joins_test, labels_test, max_num_samples, max_num_joins, max_num_predicates, sample_feats, predicate_feats, join_feats)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error " + workload_name + ":")
    qerror = print_qerror(preds_test_unnorm, label)

    # Write predictions
    file_name = f"{save_path}/predictions_" + workload_name + ".csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    print(f"Writing results to {file_name}")
    with open(file_name, "w") as f:
        f.write("Median: {}\n".format(np.median(qerror)))
        f.write("90th percentile: {}\n".format(np.percentile(qerror, 90)))
        f.write("95th percentile: {}\n".format(np.percentile(qerror, 95)))
        f.write("99th percentile: {}\n".format(np.percentile(qerror, 99)))
        f.write("Max: {}\n".format(np.max(qerror)))
        f.write("Mean: {}\n\n".format(np.mean(qerror)))
        f.write("preds, labels\n")
        for i in range(len(preds_test_unnorm)):
            f.write(str(preds_test_unnorm[i]) + "," + label[i] + "\n")


def train(num_queries, num_epochs, batch_size, hid_units, cuda, save_path, materialize, lr):
    if materialize:
        # Load training and validation data
        num_materialized_samples = 1000
    else:
        num_materialized_samples = 0
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_samples, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts
    dicts.extend([column_min_max_vals, min_val, max_val, max_num_samples, max_num_joins, max_num_predicates])
    torch.save(dicts, os.path.join(save_path, 'saved_dicts.pt'))
    # Train model
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConv(sample_feats, predicate_feats, join_feats, hid_units, max_num_samples, max_num_joins, max_num_predicates)





    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_data_loader))
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    model.train()
    for epoch in range(num_epochs):
        loss_total = 0.
        optimizer.zero_grad()
        for batch_idx, data_batch in enumerate(train_data_loader):

            inputs, targets = data_batch

            if cuda:
            #     samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            #     sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
            # samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(
            #     targets)
            # sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
            #     join_masks)
                inputs = inputs.cuda()
                inputs = Variable(inputs)
            ONNX_DIR = f'../../../Benchmarks/onnx'
            save_path = f'{ONNX_DIR}/cardinality_{hid_units}_tmp.onnx'


            outputs = model(inputs)
            inputs = inputs[:1, :, :]
            print(inputs.shape)

            torch.onnx.export(model,  # model being run
                              inputs,  # model input (or a tuple for multiple inputs)
                              save_path,  # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=12,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              output_names=['output'])  # the model's output names
            return
























            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with warmup_scheduler.dampening():
                lr_scheduler.step()

        print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))
    model.eval()
    torch.save(model.state_dict(), os.path.join(save_path, 'model-dict.pt'))

    # Get final training and validation set predictions
    preds_train, t_total = predict(model, train_data_loader, cuda)
    print("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per validation sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error training set:")
    print_qerror(preds_train_unnorm, labels_train_unnorm)

    print("\nQ-Error validation set:")
    print_qerror(preds_test_unnorm, labels_test_unnorm)
    print("")



def main():

    train(10000, 10, 1024, 128, False, "./", False, 0.01)
    train(10000, 10, 1024, 2048, False, "./", False, 0.01)

if __name__ == "__main__":
    main()
