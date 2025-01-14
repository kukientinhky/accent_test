import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from NCF.src.get_rec import get_rec
from NCF.src.helper import get_model


def main():
    """
    train a new NCF model
    """
    get_model(use_recs=False)
    get_rec()


if __name__ == "__main__":
    main()
