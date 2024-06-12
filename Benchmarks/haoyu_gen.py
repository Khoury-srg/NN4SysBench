import os
import sys
import math
import random
import numpy as np
from src.mscn.data import encode_query, normalize_labels


def is_every_leq_or_geq(x):
    flag = None
    count = 0
    flex_index = []
    for i in range(3, 9):
        if flag == 9 and x[i * 14 + 11] == 1.0:
            return False, count, flex_index
        elif flag == 11 and x[i * 14 + 9] == 1.0:
            return False, count, flex_index
        else:
            if x[i * 14 + 9] == 1.0:
                flag = 9
            elif x[i * 14 + 11] == 1.0:
                flag = 11
        if x[i * 14 + 10] != 1.0 and (x[i * 14 + 11] == 1.0 or x[i * 14 + 9] == 1.0):
            count += 1
            flex_index.append(i * 14 + 11 if x[i * 14 + 11] == 1.0 else i * 14 + 9)
    return flag if flag else False, count, flex_index


def trans_multi_dual_vnnlib(spec_path, queries, testset, safe=True):
    all_tensor = encode_query(queries, 0, 'src/mscn/mscn_resources', testset)[0].tensors[0]
    all_tensor = [i.flatten() for i in all_tensor]
    return_indexes = []
    difficulties = []

    with open(spec_path, "w") as f:
        # f.write(f"; CIFAR10 property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(all_tensor[0].shape[0] * 2):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write(f"(declare-const Y_0 Real)\n")
        if safe:
            f.write("\n(assert (or \n")
        else:
            f.write("\n(assert \n")
        for tt in range(len(all_tensor)):
            flag, count, flex_index = is_every_leq_or_geq(all_tensor[tt])
            f.write("(and ")
            for i, j in enumerate(all_tensor[tt]):
                if i in flex_index:
                    f.write(f"(>= X_{i} {round(min(j.item() + 0.0001, 1.0), 16)}) (<= X_{i} {1.0}) ")
                else:
                    f.write(f"(>= X_{i} {round(j.item(), 16)}) (<= X_{i} {round(j.item(), 16)}) ")

            for i, j in enumerate(all_tensor[tt]):
                if i in flex_index:
                    f.write(f"(>= X_{i + 154} {0.0}) (<= X_{i + 154} {round(max(0.0, j.item() - 0.0001), 16)}) ")
                else:
                    f.write(f"(>= X_{i + 154} {round(j.item(), 16)}) (<= X_{i + 154} {round(j.item(), 16)}) ")
            if flag == 9:
                f.write("(>= Y_0 1e-5))\n")
            elif flag == 11:
                f.write("(<= Y_0 -1e-5))\n")
            else:
                raise NotImplementedError
        if safe:
            f.write("))\n")
        else:
            f.write(")\n")
    print(f"[DONE] generate {spec_path}")
    return difficulties, return_indexes


def trans_multi_vnnlib(spec_path, queries, doppel_queries, label_range, testset, safe=True):
    all_tensor = encode_query(queries, 0, 'src/mscn/mscn_resources', testset)[0].tensors[0]
    all_dopple_tensor = \
        encode_query(doppel_queries, 0, 'src/mscn/mscn_resources', testset)[0].tensors[0]
    all_tensor = [i.flatten() for i in all_tensor]
    all_dopple_tensor = [i.flatten() for i in all_dopple_tensor]
    with open(spec_path, "w") as f:
        f.write("\n")
        for i in range(all_tensor[0].shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write(f"(declare-const Y_0 Real)\n")
        if safe:
            f.write("\n(assert (or \n")
        else:
            f.write("\n(assert \n")
        for _i in range(len(queries)):
            tensor = all_tensor[_i]
            dopple_tensor = all_dopple_tensor[_i]
            sub_label_range = label_range[_i]

            # for tt in range(len(tensor)):
            #     lr = sub_label_range[tt]
            for ri, r in enumerate(sub_label_range):
                if r == None:
                    continue
                f.write("(and ")
                for i, (j, k) in enumerate(zip(tensor, dopple_tensor)):
                    f.write(f"(>= X_{i} {round(min(j, k).item(), 16)}) (<= X_{i} {round(max(j, k).item(), 16)}) ")

                if ri == 0:
                    f.write(f"(<= Y_0 {round(normalize_labels([r], 0.0, 19.94772801931604)[0][0], 16)}))\n")
                else:
                    f.write(f"(>= Y_0 {round(normalize_labels([r], 0.0, 19.94772801931604)[0][0], 16)}))\n")
        if safe:
            f.write("))\n")
        else:
            f.write(")\n")
    print(f"[DONE] generate {spec_path}")


def gene_spec():
    dual_queries = open('src/mscn/mscn_resources/dual_query.txt').read().splitlines()
    queries = open('src/mscn/mscn_resources/query.txt').read().splitlines()
    dopple_queries = open('src/mscn/mscn_resources/dopple_query.txt').read().splitlines()
    label_range = open('src/mscn/mscn_resources/label_range.txt').read().splitlines()
    label_range = [(int(i.split(',')[0]), int(i.split(',')[1])) for i in label_range]
    time_dict = np.load('src/mscn/mscn_resources/time_dict.npy', allow_pickle=True).item()
    time_dict_single = np.load('src/mscn/mscn_resources/time_dict_single.npy', allow_pickle=True).item()

    csv_data = []
    dir_path = 'vnnlib'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    single_difficulties = [1]
    single_difficulties.extend([(i + 1) * 250 for i in range(1, 6)])
    single_difficulties.extend([(i + 1) * 560 + 1000 for i in range(1, 6)])
    for model in ['128', '2048']:
        # generate single instances
        index=-1
        for size in single_difficulties:
            index+=1
            for sf, indexes in enumerate(np.load(f'src/mscn/mscn_resources/sf_list_{model}.npy', allow_pickle=True)):
                if sf == 0:
                    continue
                # chosen_dmap = {0: [i for i in indexes if difficulties[i]=='0'],
                #                1: [i for i in indexes if difficulties[i]=='1']}

                # for difficulty in [0, 1]:
                try:
                    # chosen_index = random.sample(chosen_dmap[difficulty], size)
                    chosen_index = random.sample(indexes, size)
                except:
                    continue
                trans_multi_vnnlib(f'{dir_path}/mscn_{model}d_{index}.vnnlib',
                                   [queries[i] for i in chosen_index], [dopple_queries[i] for i in chosen_index],
                                   [label_range[i] for i in chosen_index], 'scale',
                                   safe=bool(sf))
                csv_data.append([f'onnx/mscn_{model}d.onnx',
                                 f'vnnlib/mscn_{model}d_{index}.vnnlib',
                                 max(time_dict_single['single'][size] // 2, 20) if model == '128' else
                                 time_dict_single['single'][size]])
        # generate dual instances
        dual_difficulties = [1]
        dual_difficulties.extend([(i + 1) * 120 for i in range(1, 6)])
        dual_difficulties.extend([(i + 1) * 630 + 1000 for i in range(1, 6)])
        index=-1
        for size in dual_difficulties:
            index+=1
            for sf, indexes in enumerate(
                    np.load(f'src/mscn/mscn_resources/sf_list_{model}_dual.npy', allow_pickle=True)):
                if sf == 0:
                    continue
                try:
                    chosen_index = random.sample(indexes, size)
                except:
                    continue
                trans_multi_dual_vnnlib(f'{dir_path}/mscn_{model}d_dual_{index}.vnnlib',
                                        [dual_queries[i] for i in chosen_index], 'scale', bool(sf))
                csv_data.append([f'onnx/mscn_{model}d_dual.onnx',
                                 f'vnnlib/mscn_{model}d_dual_{index}.vnnlib',
                                 time_dict[int(model)][size]])
    return csv_data


def main(seed):
    random.seed(seed)
    return gene_spec()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_properties.py <random seed>")
        #exit(1)
        random_seed = 2024
    else:
        random_seed = sys.argv[1]
    main(random_seed)
