import os
import math
import numpy as np
import sys


def calculate_avg_time(dic1, dic2, times1, times2):
    ret = {}
    for key in times2:
        if key in times1:
            times1[key] += times2[key]
        else:
            times1[key] = times2[key]
    for key in dic2:
        if key in dic1:
            dic1[key] += dic2[key]
        else:
            dic1[key] = dic2[key]
    for key in dic1:
        if dic1[key]==10:
            ret[key] = times1[key] / dic1[key]
        else:
            timeout = 10-dic1[key]
            total_time = timeout*180+times1[key]
            ret[key] = total_time / 10

    return ret


def main(model):
    dir = f'./{model}_marabou_running_result'
    files = os.listdir(dir)

    unsat = 0
    sat = 0
    sat_dic = {}
    unsat_dic = {}
    sat_time = {}
    unsat_time = {}
    print("sat files:----------------------------------------------------------")

    for f in files:
        file = f'{dir}/' + f
        if file[-3:] != 'txt':
            continue
        index = '_'.join(f[:-4].split('_')[:-1])
        timeout = -1

        with open(file, 'r') as f:
            result = ''
            for line in f:
                line = line.strip()
                if line[:3] == "sat":
                    result = "sat"
                if line[:5] == "unsat":
                    result = "unsat"
                if line[:4] == "Time":
                    timeout = float(line[5:15])

            if timeout == -1:
                continue
            timeout = float(timeout)
            if result == 'unsat':
                unsat += 1
                if index in unsat_dic.keys():
                    unsat_dic[index] = unsat_dic[index] + 1
                    unsat_time[index] = unsat_time[index] + timeout
                else:
                    unsat_dic[index] = 1
                    unsat_time[index] = timeout

            elif result == 'sat':
                sat += 1
                if index in sat_dic.keys():
                    sat_dic[index] = sat_dic[index] + 1
                    sat_time[index] = sat_time[index] + timeout
                else:
                    sat_dic[index] = 1
                    sat_time[index] = timeout

            else:
                print("no result")
                print(file)
    print("----------------------------------------------------------sat files")
    print(f'sat: {sat}')

    print(f'unsat: {unsat}')

    sat_dic = dict(sorted(sat_dic.items()))
    unsat_dic = dict(sorted(unsat_dic.items()))
    print("sat")
    print(len(sat_dic))
    print(sat_dic)

    print("unsat")
    print(len(unsat_dic))
    print(unsat_dic)

    avg_time = calculate_avg_time(sat_dic, unsat_dic, sat_time, unsat_time)
    print(avg_time)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python abcrown_parser.py model")
        exit(1)
    else:
        model = sys.argv[1].lower()

    main(model)
