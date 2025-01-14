import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from NCF.src.get_counterfactual import generate_cf
from NCF.src.get_new_scores import get_new_scores
from NCF.src.helper import parse_args
from NCF.src.retrain import retrain
from commons.helper import evaluate_files


def main():
    """
    run the full experiment for an algorithm passed via the command line argument --algo
    """
    args = parse_args()
    ks = [5, 10, 20]
    generate_cf(args.algo, ks)
    retrain(args.algo, ks)
    get_new_scores(args.algo, ks)
    evaluate_files(parse_args, ks)


if __name__ == "__main__":
    main()
