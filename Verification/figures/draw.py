import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib

import copy
import os
import math
import numpy as np
import sys
import json


def adjust_map(map):
    # find the largest value
    max_length = 0
    for key, value in map.items():
        max_length = max(len(value), max_length)

    # modify all value
    for key, value in map.items():
        length = len(value)

        delta = max_length - length

        left = int(delta / 2)
        right = delta - left

        left_space = ""
        for i in range(left):
            left_space = left_space + " "
        right_space = ""

        for i in range(right):
            right_space = right_space + " "

        map[key] = right_space + value + left_space

    return map


def main():
    map = {'aurora_big_101': "CongestCtrl_large_spec101",
           'aurora_big_102': "CongestCtrl_large_spec102",
           'aurora_big_2': "CongestCtrl_large_spec2",
           'aurora_big_3': "CongestCtrl_large_dual_spec3",
           'aurora_big_4': "CongestCtrl_large_chain_spec4",

           'aurora_mid_101': "CongestCtrl_mid_spec101",
           'aurora_mid_102': "CongestCtrl_mid_spec102",
           'aurora_mid_2': "CongestCtrl_mid_spec2",
           'aurora_mid_3': "CongestCtrl_mid_dual_spec3",
           'aurora_mid_4': "CongestCtrl_mid_chain_spec4",

           'aurora_small_101': "CongestCtrl_small_spec101",
           'aurora_small_102': "CongestCtrl_small_spec102",
           'aurora_small_2': "CongestCtrl_small_spec2",
           'aurora_small_3': "CongestCtrl_small_dual_spec3",
           'aurora_small_4': "CongestCtrl_small_chain_spec4",

           'bloom_filter': "BloomFilter",
           'decima_mid_1': "LearnedSched_spec1",
           'decima_mid_2': "LearnedSched_spec2",
           'lindex_0': "LearnedIndex_spec1",
           'lindex_1': "LearnedIndex_spec2",
           'lindex_2': "LearnedIndex_spec3",
           'lindex_deep_0': "LearnedIndex_large_spec1",
           'lindex_deep_1': "LearnedIndex_large_spec2",
           'lindex_deep_2': "LearnedIndex_large_spec3",
           'mscn_128d': "CardEsti_small",
           'mscn_128d_dual': "CardEsti_small_dual",
           'mscn_2048d': "CardEsti_large",
           'mscn_2048d_dual': "CardEsti_large_dual",

           'pensieve_big_1': "AdaptBitrate_large_spec1",
           'pensieve_big_2': "AdaptBitrate_large_spec2",
           'pensieve_big_3': "AdaptBitrate_large_dual_spec3",
           'pensieve_mid_1': "AdaptBitrate_mid_spec1",
           'pensieve_mid_2': "AdaptBitrate_mid_spec2",
           'pensieve_mid_3': "AdaptBitrate_mid_dual_spec3",
           'pensieve_small_1': "AdaptBitrate_small_spec1",
           'pensieve_small_2': "AdaptBitrate_small_spec2",
           'pensieve_small_3': "AdaptBitrate_small_dual_spec3",
           }

    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'normal',
            'size': 15
            }
    font_text = {'family': 'Times New Roman',
                 'color': 'black',
                 'weight': 'normal',
                 'size': 14
                 }
    font_legend = {'family': 'Times New Roman', 'size': 15}

    f = open('eval_results.json')
    content = f.read()
    datas = json.loads(content)
    f.close()

    species = []

    for key in datas:
        species.append(key)

    # first table
    content1 = "\mr{{2}}*{{\\ttt{{{%s}}}}} & \\ab & %s & %s & %s \\\\ \n                                & \ma & %s & %s & %s \\\\"
    content2 = "\mr{{1}}*{{\\ttt{{{%s}}}}} & \\ab & %s & %s & %s \\\\ \n                                & \ma & - & - & - \\\\"

    for key in datas:
        string = map[key]
        string = string.replace("_", "\\_")
        if "marabou" in datas[key].keys():
            map_ab = datas[key]["abcrown"]
            map_ma = datas[key]["marabou"]
            print(content1 % (string,str(map_ab["safe"]), str(map_ab["unsafe"]), str(map_ab["timeout"]),str(
                map_ma["safe"]), str(map_ma["unsafe"]), str(map_ma["timeout"])))
        else:
            map_ab = datas[key]["abcrown"]
            print(content2 % (string, str(map_ab["safe"]),str(map_ab["unsafe"]),str(map_ab["timeout"])))

    # second fiture: runtime

    marabou_can_dic_for_sort = {}
    marabou_cannot_dic_for_sort = {}

    for key in datas:
        if 'marabou' in datas[key]:
            marabou_can_dic_for_sort[key] = datas[key]['abcrown']['time']
        else:
            marabou_cannot_dic_for_sort[key] = datas[key]['abcrown']['time']
    sort = sorted(marabou_can_dic_for_sort.items(), key=lambda x: x[1])
    marabou_can_dic_for_sort = dict(sort)
    sort = sorted(marabou_cannot_dic_for_sort.items(), key=lambda x: x[1])
    marabou_cannot_dic_for_sort = dict(sort)

    time_sets = {}
    annotates = {}

    time_sets['abcrown'] = []
    time_sets['marabou'] = []
    annotates['abcrown'] = []
    annotates['marabou'] = []

    x_ticks = []

    for key in marabou_can_dic_for_sort:
        time_sets['abcrown'].append(int(datas[key]['abcrown']['time'] * 100) / 100.0)
        time_sets['marabou'].append(int(datas[key]['marabou']['time'] * 100) / 100.0)
        annotates['abcrown'].append(int(datas[key]['abcrown']['time'] * 100) / 100.0)
        annotates['marabou'].append(int(datas[key]['marabou']['time'] * 100) / 100.0)
        x_ticks.append(key)

    for key in marabou_cannot_dic_for_sort:
        time_sets['abcrown'].append(int(datas[key]['abcrown']['time'] * 100) / 100.0)
        time_sets['marabou'].append(0)
        annotates['abcrown'].append(int(datas[key]['abcrown']['time'] * 100) / 100.0)
        annotates['marabou'].append('NaN')
        x_ticks.append(key)

    x = np.arange(len(species))  # the label locations
    width = 0.40  # the width of the bars
    fig, ax = plt.subplots(layout='constrained', figsize=(16, 4.5))

    plt.bar(x + width / 2, time_sets['abcrown'], width, label=chr(945) + chr(946) + '-CROWN', edgecolor="black",
            linewidth=2)
    time_sets['abcrown'][-1] += 0  # adjust the text y position so that texts will not overlap
    time_sets['abcrown'][-2] += 160  # adjust the text y position so that texts will not overlap
    time_sets['abcrown'][-3] += 30  # adjust the text y position so that texts will not overlap
    time_sets['abcrown'][17] += 1
    time_sets['abcrown'][18] += 25
    time_sets['abcrown'][14] += 1
    [plt.text(i + width / 2, time_sets['abcrown'][i], str(annotates['abcrown'][i]), ha='center', va='bottom',
              fontdict=font_text) for i in range(len(x_ticks))]

    ax.bar(x + width / 2 + width, time_sets['marabou'], width, label='Marabou', edgecolor="black", linewidth=2)
    time_sets['marabou'][17] += 15  # adjust the text y position so that texts will not overlap
    [plt.text(i + width / 2 + width, max(time_sets['marabou'][i], 0.013), str(annotates['marabou'][i]), ha='center',
              va='bottom', fontdict=font_text) for i in range(len(x_ticks))]

    plt.xlim(left=-0.5, right=37)
    ax.legend(loc='upper left', ncols=3, prop=font_legend)
    ax.set_xticks(x + width, [map[i] for i in x_ticks], fontdict=font)

    ax.set_ylabel('Average Runtime (s)', fontdict=font)

    plt.tick_params(labelsize=15, labelfontfamily='Times New Roman')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    [x.set_linewidth(2) for x in ax.spines.values()]

    plt.xticks(rotation=-30, ha="left", rotation_mode="anchor")
    plt.yscale("log")
    plt.savefig("verification_runtime_2.pdf", format="pdf", bbox_inches="tight")

    # figure 3
    fig, ax = plt.subplots(layout='constrained', figsize=(16, 4.5))
    sizes = []
    runtime = []

    size_dic = {}
    run_time_dic = {}

    for key in datas:
        if not "pensieve" in key and not "aurora" in key:
            continue
        if 'size' in datas[key]:
            size_dic[key] = datas[key]['size']
            run_time_dic[key] = datas[key]["abcrown"]["time"]
    size_list = sorted(size_dic.items(), key=lambda x: x[1])
    size_dic = dict(size_list)

    for key in size_dic:
        runtime.append(run_time_dic[key])

    width = 0.7

    names = [map[i] for i in size_dic.keys()]
    l1 = ax.bar(names, runtime, width=width, label='Verification Runtime', lw=2, ec='black')
    plt.xticks(rotation=-30, ha="left", rotation_mode="anchor", fontproperties='Times New Roman', size=15)
    position = copy.deepcopy(runtime)
    position[10] += 0.4  # adjust the text y position so that texts will not overlap
    position[12] += 0.3  # adjust the text y position so that texts will not overlap
    position[13] += 0.4  # adjust the text y position so that texts will not overlap
    for i in range(len(size_dic.keys())):
        texts = plt.text(names[i], position[i], str(int(100 * runtime[i]) / 100.0), ha='center', va='bottom',
                         fontdict=font_text)
    plt.tick_params(labelsize=15, labelfontfamily='Times New Roman')

    ax2 = ax.twinx()
    ax2.plot(names, size_dic.values(), '-o', label='Model Size', color='orange')
    plt.xlim(left=-1, right=24)

    # ax.set_title('Verification Runtime and Model Size')
    ax2.set_ylabel('Size (byte)', fontdict=font)
    ax.set_ylabel('Average Runtime (s)', fontdict=font)
    plt.yscale("log")
    plt.tick_params(labelsize=15, labelfontfamily='Times New Roman')

    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    fig.legend(loc=(0.04, 0.9), ncols=3, prop=font_legend)
    [x.set_linewidth(2) for x in ax.spines.values()]
    plt.savefig("verification_runtime_and_model_size_2.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
