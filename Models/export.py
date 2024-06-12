import argparse

import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="all",
                        choices=["pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"],
                        help="which model to verify pensieve, decima, lindex, cardinality, bloom_filter, aurora")

    args = parser.parse_args()
    if args.model == "all":
        os.system("cd Pensieve && python export.py && cd .. && cd Decima && python export.py && cd .."
                  "&& cd Learned_index && python export.py && cd .."
                  "&& cd Cardinality && python export.py && cd .."
                  "&& cd Bloom_filter && python export.py && cd .."
                  "&& cd Aurora && python export.py && cd ..")

    if args.model == "pensieve":
        os.system("cd Pensieve && python export.py && cd ..")

    if args.model == "decima":
        os.system("cd Decima && python export.py && cd ..")

    if args.model == "lindex":
        os.system("cd Learned_index && python export.py && cd ..")

    if args.model == "cardinality":
        os.system("cd Cardinality && python export.py && cd ..")

    if args.model == "bloom_filter":
        os.system("cd Bloom_filter && python export.py && cd ..")

    if args.model == "aurora":
        os.system("cd Aurora && python export.py && cd ..")



if __name__ == "__main__":
    main()
