from RCF.src.generate_counterfactual import generate_cf
from RCF.src.helper import parse_args
from commons.helper import evaluate_files
from get_new_scores import get_new_scores
from retrain_counterfactual import retrain
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def main():
    """
    run the full experiment for an algorithm passed via the command line argument --algo
    """
    ks = [5, 10, 20]
    generate_cf(ks)
    print("-----------------------------------")
    print("Retraining the model")
    print("--------------------------------")
    retrain(ks)
    print("-----------------------------------")
    print("Get new scores")
    print("--------------------------------")
    get_new_scores(ks)
    print("-----------------------------------")
    print("Evaluating the model")
    print("--------------------------------")
    evaluate_files(parse_args, ks)


if __name__ == "__main__":
    main()
