import argparse

import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="all",
                        choices=["pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"],
                        help="which model to generate pensieve, decima, lindex, cardinality, bloom_filter, aurora")

    args = parser.parse_args()
    if args.model == "all":
        os.system("cd Pensieve && python shuyi_gen_upper.py && cd .. "
                  "&& cd Decima && python shuyi_gen_upper.py && cd .."
                  "&& cd Aurora && python shuyi_gen_upper.py && cd ..")

    if args.model == "pensieve":
        os.system("cd Pensieve && python shuyi_gen_upper.py && cd ..")

    if args.model == "decima":
        os.system("cd Decima && python shuyi_gen_upper.py && cd ..")


    if args.model == "aurora":
        os.system("cd Aurora && python shuyi_gen_upper.py && cd ..")


if __name__ == "__main__":
    main()
