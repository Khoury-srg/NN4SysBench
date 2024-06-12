import argparse
import Aurora_marabou_run
import Pensieve_marabou_run
import Bloom_filter_marabou_run
import Lindex_marabou_run
import Decima_marabou_run


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="all",
                        choices=["pensieve", "decima", "lindex", "bloom_filter", "aurora", "all"],
                        help="which model to verify pensieve, decima, lindex, bloom_filter, aurora")
    parser.add_argument("--path",
                        help="verifier path")

    args = parser.parse_args()
    if args.model == "all":
        Pensieve_marabou_run.main(args.path)
        Decima_marabou_run.main(args.path)
        Lindex_marabou_run.main(args.path)
        Bloom_filter_marabou_run.main(args.path)
        Aurora_marabou_run.main(args.path)
    if args.model == "pensieve":
        Pensieve_marabou_run.main(args.path)
    if args.model == "decima":
        Decima_marabou_run.main(args.path)
    if args.model == "lindex":
        Lindex_marabou_run.main(args.path)
    if args.model == "bloom_filter":
        Bloom_filter_marabou_run.main(args.path)
    if args.model == "aurora":
        Aurora_marabou_run.main(args.path)


if __name__ == "__main__":
    main()
