import copy
import os
import json

Models = ['pensieve', 'lindex', 'aurora', 'decima', 'bloom_filter', 'cardinality']
Verifiers = ['abcrown', 'marabou']

ONND_DIR = "../../Benchmarks/onnx/"

spec_to_model_map = {"aurora_big_101": "aurora_big_simple.onnx",
                     "aurora_big_102": "aurora_big_simple.onnx",
                     "aurora_big_2": "aurora_big_simple.onnx",
                     "aurora_big_3": "aurora_big_parallel.onnx",
                     "aurora_big_4": "aurora_big_concat.onnx",

                     "aurora_small_101": "aurora_small_simple.onnx",
                     "aurora_small_102": "aurora_small_simple.onnx",
                     "aurora_small_2": "aurora_small_simple.onnx",
                     "aurora_small_3": "aurora_small_parallel.onnx",
                     "aurora_small_4": "aurora_small_concat.onnx",

                     "aurora_mid_101": "aurora_mid_simple.onnx",
                     "aurora_mid_102": "aurora_mid_simple.onnx",
                     "aurora_mid_2": "aurora_mid_simple.onnx",
                     "aurora_mid_3": "aurora_mid_parallel.onnx",
                     "aurora_mid_4": "aurora_mid_concat.onnx",

                     "decima_mid_1": "decima_mid_simple.onnx",
                     "decima_mid_2": "decima_mid_simple.onnx",

                     "pensieve_big_1": "pensieve_big_simple.onnx",
                     "pensieve_big_2": "pensieve_big_simple.onnx",
                     "pensieve_big_3": "pensieve_big_parallel.onnx",
                     "pensieve_small_1": "pensieve_small_simple.onnx",
                     "pensieve_small_2": "pensieve_small_simple.onnx",
                     "pensieve_small_3": "pensieve_small_parallel.onnx",
                     "pensieve_mid_1": "pensieve_mid_simple.onnx",
                     "pensieve_mid_2": "pensieve_mid_simple.onnx",
                     "pensieve_mid_3": "pensieve_mid_parallel.onnx",

                     "mscn_128d": "mscn_128d.onnx",
                     "mscn_128d_dual": "mscn_128d_dual.onnx",
                     "mscn_2048d_dual": "mscn_2048d_dual.onnx",
                     "mscn_2048d": "mscn_2048d.onnx",

                     "lindex_0": "lindex.onnx",
                     "lindex_1": "lindex.onnx",
                     "lindex_2": "lindex.onnx",

                     "linde_deep_0": "lindex_deep.onnx",
                     "linde_deep_1": "lindex_deep.onnx",
                     "linde_deep_2": "lindex_deep.onnx"

                     }


def calculate_avg_time(dic1, dic2, dic3, times1, times2, time3):


    ret = {}
    for key in times2:
        if key in times1:
            times1[key] += times2[key]
        else:
            times1[key] = times2[key]
    for key in time3:
        if key in times1:
            times1[key] += time3[key]
        else:
            times1[key] = time3[key]
    for key in dic2:
        if key in dic1:
            dic1[key] += dic2[key]
        else:
            dic1[key] = dic2[key]
    for key in dic3:
        if key in dic1:
            dic1[key] += dic3[key]
        else:
            dic1[key] = dic3[key]
    for key in dic1:
        if dic1[key] == 10:
            ret[key] = times1[key] / dic1[key]
        else:
            timeout = 10 - dic1[key]
            total_time = timeout * 180 + times1[key]
            ret[key] = total_time / 10

    return ret


def init_dic():
    ret = {}

    ret['safe'] = 0
    ret['unsafe'] = 0
    ret['time'] = 0
    ret['timeout'] = 0

    return ret


def main():
    datas = {}
    for verifier in Verifiers:
        for model in Models:
            dir = f'../{model}_{verifier}_running_result'
            if not os.path.exists(dir):
                continue
            files = os.listdir(dir)


            unsat = 0
            sat = 0
            sat_dic = {}
            unsat_dic = {}
            sat_time = {}
            unsat_time = {}
            timeout_dic = {}
            timeout_time = {}

            for f in files:

                file = f'{dir}/' + f
                if file[-3:] != 'txt':
                    continue
                index = '_'.join(f[:-4].split('_')[:-1])
                if 'pensieve' in index or 'aurora' in index:
                    index = '_'.join(f.split('_')[:-3])



                timeout = -1

                with open(file, 'r') as f:
                    result = 'timeout'
                    if verifier == 'abcrown':
                        for line in f:
                            line = line.strip()
                            if line[:6] == "Result":
                                result = line[8:]
                            if line[:4] == "Time":
                                timeout = float(line[6:15])
                    else:
                        for line in f:
                            line = line.strip()
                            if line[:3] == "sat":
                                result = "sat"
                            if line[:5] == "unsat":
                                result = "unsat"
                            if line[:4] == "Time":
                                timeout = float(line[5:15])

                    if timeout > 180:
                        result = "timeout"
                        timeout = 180


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
                        if index in timeout_dic.keys():
                            timeout_dic[index] = timeout_dic[index] + 1
                            timeout_time[index] = timeout_time[index] + 180
                        else:
                            timeout_dic[index] = 1
                            timeout_time[index] = 180

            sat_dic = dict(sorted(sat_dic.items()))
            sat_dic_copy = copy.deepcopy(sat_dic)
            unsat_dic = dict(sorted(unsat_dic.items()))
            unsat_dic_copy = copy.deepcopy(unsat_dic)
            timeout_dic_copy = copy.deepcopy(timeout_dic)

            avg_time = calculate_avg_time(sat_dic_copy, unsat_dic_copy, timeout_dic_copy, sat_time, unsat_time,
                                          timeout_time)


            for key in avg_time:
                if not key in datas:
                    datas[key] = {}
                if not verifier in datas[key]:
                    datas[key][verifier] = init_dic()

                datas[key][verifier]['time'] = avg_time[key]

                if key in sat_dic:
                    datas[key][verifier]['unsafe'] = sat_dic[key]
                if key in unsat_dic:
                    datas[key][verifier]['safe'] = unsat_dic[key]
                if key in timeout_dic:
                    datas[key][verifier]['timeout'] = timeout_dic[key]
                else:
                    datas[key][verifier]['timeout'] = 10 - datas[key][verifier]['unsafe'] - datas[key][verifier]['safe']



            for key in datas:
                try:
                    size = os.path.getsize(ONND_DIR + spec_to_model_map[key])
                    datas[key]["size"] = size
                except:
                    continue
    for key in datas.keys():
        print(key)
        print(datas[key])
    datas = dict(sorted(datas.items()))

    datas = json.dumps(datas)
    f = open('eval_results.json', 'w')
    f.write(datas)
    f.close()


if __name__ == "__main__":
    main()
