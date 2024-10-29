import math
import sys
import random

spec_path = "src/spec_all.txt"


def load_spec():
    with open(spec_path, "r") as f:
        content = f.read()

    specs = []
    lines = content.split()
    for line in lines:
        if len(line) == 0:  # skip
            continue
        tokens = line.split(",")
        assert len(tokens) == 4
        specs.append([float(x) for x in tokens])

    assert len(specs) > 10000
    return specs


def specs_shift(specs, shift):
    new_specs = []

    for i in range(shift, len(specs)):
        assert len(specs[i]) == 4
        new_specs.append([specs[i][0], specs[i][1],
                          specs[i - shift][2], specs[i - shift][3]])

    return new_specs


def write_spec(path, spec_set):
    contents = ["(declare-const X_0 Real)",
                "(declare-const Y_0 Real)",
                "(assert (or"]
    lower_template = "  (and (>= X_0 %.16f) (<= X_0 %.16f) (<= Y_0 %.16f))"
    upper_template = "  (and (>= X_0 %.16f) (<= X_0 %.16f) (>= Y_0 %.16f))"

    for spec in spec_set:
        x_lb, x_ub, y_lb, y_ub = spec
        contents.append(lower_template % (x_lb, x_ub, y_lb))
        contents.append(upper_template % (x_lb, x_ub, y_ub))

    contents.append("))")

    with open(path, "w") as f:
        f.write("\n".join(contents))


def write_uspec(path, spec_set):
    contents = ["(declare-const X_0 Real)",
                "(declare-const Y_0 Real)"
                ]
    lower_template = "assert(and (>= X_0 %.16f) (<= X_0 %.16f) (<= Y_0 %.16f))"
    upper_template = "assert(and (>= X_0 %.16f) (<= X_0 %.16f) (>= Y_0 %.16f))"

    for spec in spec_set:
        x_lb, x_ub, y_lb, y_ub = spec
        contents.append(lower_template % (x_lb, x_ub, y_lb))
        contents.append(upper_template % (x_lb, x_ub, y_ub))

    with open(path, "w") as f:
        f.write("\n".join(contents))

def write_txt(path, spec_set):
    contents = []


    for spec in spec_set:
        x_lb, x_ub, y_lb, y_ub = spec
        contents.append(f"x0 >= {x_lb}")
        contents.append(f"x0 <= {x_ub}")
        contents.append(f"y0 <= {y_lb}")
        contents.append(f"y0 >= {y_ub}")


    with open(path, "w") as f:
        f.write("\n".join(contents))
    print(f"[Done] generate {path}")

def gen_spec(specs, difficulties, dif):
    assert len(difficulties) > 0
    num_specs = len(specs)

    ftemplate = "vnnlib/lindex_%d_%d.vnnlib"
    ftemplate2 = "marabou_txt/lindex_%d_%d.txt"
    all_names = []
    csv_data = []
    index=-1
    for num in difficulties:
        index+=1
        assert num <= num_specs
        chosen_ids = random.sample(range(num_specs), num)

        chosen_ones = []
        for spec_id in chosen_ids:
            chosen_ones.append(specs[spec_id])

        fname = ftemplate % (dif, index)
        write_spec(fname, chosen_ones)
        if dif==0:
            write_txt(ftemplate2 % (dif, index), chosen_ones)
        csv_data.append(['onnx/lindex.onnx', fname, 20 + math.ceil(num * 3 / num_specs) * 10])
        csv_data.append(['onnx/lindex_deep.onnx', fname, 20 + math.ceil(num * 3 / num_specs) * 10])
        print(f"[DONE] generate {fname}")
        all_names.append(fname)

    # write spec names to "spec/lindex_files_names.txt"
    with open("vnnlib/lindex_files_names.txt", "w") as f:
        f.write("\n".join(all_names))
    return csv_data


def gen_uspec(uspecs):
    num_uspecs = len(uspecs)
    csv_data = []
    # three difficulties
    difficulties = [1, 100, 10000]

    # generate unsafe specs
    ftemplate = "vnnlib/ulindex_%d.vnnlib"
    for num in difficulties:
        chosen_ones = []
        for i in range(num):
            chosen_ones.append(uspecs[random.randint(0, num_uspecs - 1)])

        fname = ftemplate % num
        write_uspec(fname, chosen_ones)
        print(f"[DONE] generate {fname}")



def main(seed, size):
    random.seed(seed)
    # load specs
    specs = load_spec()
    # print(len(specs))

    # three difficulties
    difficulties1 = [1 for i in range(10)]
    # 100 instances: 20-30-50
    # (a) easy [1, 1k] (20 instances)
    #for i in range(1, 4):
    #    difficulties.append(i * 200)  # max: 900

    # (b) medium  [1K, 10K] (30 instances)
    difficulties2=[]
    for i in range(10):
        difficulties2.append(1000 + i * 1000)  # max: 9700

    # (c) hard [10K, 144K] (50 instances)
    difficulties3=[]
    for i in range(10):
        difficulties3.append(10000 + i * 13000)  # max: 147200
    gen_spec(specs, difficulties1,0)
    gen_spec(specs, difficulties2,1)
    gen_spec(specs, difficulties3,2)
    #return gen_spec(specs, difficulties1)

    # uspecs = specs_shift(specs, 500)
    # gen_uspec(uspecs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_properties.py <random seed>")
        #exit(1)
        random_seed = 2024
    else:
        random_seed = sys.argv[1]
    main(random_seed)