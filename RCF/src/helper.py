import argparse
from collections import Counter

from tensorflow_addons.activations import gelu

from RCF.src.Utilis import get_relational_data


def parse_args():
    """
	parse all args from the console
	Returns:
		the parsed args
	"""
    parser = argparse.ArgumentParser(description="Run MF for ML.")

    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the '
                             'model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0.0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, '
                             'MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--layers', nargs='?', default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[1,1]', help='Keep probability (i.e., 1-dropout_ratio) '
                                                                        'for each deep layer and the '
                                                                        'Bi-Interaction layer. 1: no dropout. Note '
                                                                        'that the last index is for the '
                                                                        'Bi-Interaction layer.')
    parser.add_argument('--attention_size', type=int, default=32,
                        help='dimension of attention_size (default: 32)')
    parser.add_argument('--alpha', type=int, default=0.5,
                        help='smoothing factor of softmax')
    parser.add_argument('--reg_t', type=float, default=0.01,
                        help='regulation for translation relational data')
    parser.add_argument('--algo', default='accent',
                        help='explanation algorithms: attention, pure_att, fia, pure_fia, gap_red, accent')
    return parser.parse_args()


def get_pretrained_RCF_model(data, args, path):
    """
	load a pretrained RCF model from disk
	Args:
		data: the dataset used for training, see dataset.py
		args: extra arguments for the model
		path: the path that stores that pretrained model

	Returns:
		the model
	"""
    activation_function = gelu
    save_file = '%s/%s_%d' % (path, 'ml1M', args.hidden_factor)
    args.pretrain = 1  # must load pretrained
    from RCF.src.rcf import MF
    model = MF(data.num_users, data.num_items, data.num_genres, data.num_directors, data.num_actors,
               data.train_data.shape[0], data.rel_data.shape[0], args.pretrain,
               args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer, args.verbose,
               eval(args.layers), activation_function, eval(args.keep_prob), save_file, args.attention_size, args.reg_t)
    return model


def get_new_RCF_model(data, args, save_file):
    """
	get a new RCF model with all default params
	Args:
		data: the dataset used for training, see dataset.py
		args: extra arguments for the model
		save_file: the path to save the new model

	Returns:
		the model
	"""
    activation_function = gelu
    if args.pretrain == 1:  # do not load pretrained
        args.pretrain = 0
    from RCF.src.rcf import MF
    model = MF(data.num_users, data.num_items, data.num_genres, data.num_directors, data.num_actors,
               data.train_data.shape[0], data.rel_data.shape[0], args.pretrain,
               args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer, args.verbose,
               eval(args.layers), activation_function, eval(args.keep_prob), save_file, args.attention_size, args.reg_t)
    return model


def get_topk(scores: list, visited: set, k: int):
    """
	given the scores, get top k recommendations
	Args:
		scores: list
		visited: list of interacted items
		k: number of items to return

	Returns:
		dict from item to score,
		top k items
	"""
    scores = Counter({idx: val for idx, val in enumerate(scores) if idx not in visited})
    topk = scores.most_common(k)
    return scores, topk


def get_item_weights(user_id, item_id, model, data, args):
    """
    get attention weights of all interactions made by user
    Args:
        user_id: ID of user
        item_id: ID of item whose score is being predicted
        model: the recommender model
        data: dataset used to trained model, see dataset.py
        args: extra arguments for model

    Returns:
        list of actions made by user, sorted by decreasing attention weights
    """
    first_att = model.get_attention_type_user(user_id)[0]

    n_types = 4
    related_items = [None] * n_types
    values = [0] * n_types
    cnt = [0] * n_types

    related_items[0], related_items[1], related_items[2], related_items[3], values[1], values[2], values[3], \
    cnt[0], cnt[1], cnt[2], cnt[3] = get_relational_data(user_id, item_id, data)
    tmp = model.get_attention_user_item(user_id, item_id, related_items, values, cnt, args)
    second_att = [tmp[0][0][0], tmp[1][0][0], tmp[2][0][0], tmp[3][0][0]]

    item_att = {}
    for i in range(n_types):
        second_att[i] = second_att[i] * first_att[i]
        for j in range(cnt[i]):
            k = related_items[i][j]
            cur = item_att.get(k, 0)
            item_att[k] = cur + second_att[i][j]

    return sorted(item_att.items(), key=lambda x: -x[1])


def init_explanation(user_id, k, model, data, args):
    """
    initialize explanations for a user
    :param user_id: id of the user whose recommendation is to be explained
    :param k: number of items to be considered for replacement
    :param model: the trained recommender model
    :param data: the dataset used to train the model
    :param args: extra arguments
    :return: current scores of each item, the top recommendation, the top-k items, attention weights of items,
    score gap between the top recommendation and the second one.
    """
    cur_scores = model.get_scores_per_user(user_id, data, args)
    visited = set(data.user_positive_list[user_id])  # get positive list for the userID
    _, topk = get_topk(cur_scores, visited, k)
    recommended_item = topk[0][0]

    item_weights = get_item_weights(user_id, recommended_item, model, data, args)
    assert (len(visited) == len(item_weights))
    cur_diff = topk[0][1] - topk[1][1]
    return cur_scores, recommended_item, [item for item, _ in topk], item_weights, cur_diff
