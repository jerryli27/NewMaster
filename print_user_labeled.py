import server_util
import pickle
import os
import argparse
import csv
import numpy as np

parser = argparse.ArgumentParser(description='Database debugging tool.')
parser.add_argument('--detailed', '-dt', type=bool, default=True,
                    help='Whether the program also prints out detailed information about which sentence does the user '
                         'labeled as well as any related comments.')
parser.add_argument('--output_path', '-o', default='./al_hand_labeled_bak/database.csv',
                    help='output_path')
parser.add_argument('--db_path', '-db', default='./al_hand_labeled_bak/database.pkl',
                    help='db_path')
args = parser.parse_args()

if __name__ == "__main__":
    path = args.db_path

    database = server_util.load_pickle(path)
    usernames = sorted(list(database.usernames))
    for user in usernames:
        print("user %s labeled %d instances." %(user, database.get_user_num_labeled(user)))

    print("There is a total of %d users." %len(usernames))

    if args.detailed:
        if not os.path.exists(os.path.dirname(args.output_path)):
            os.mkdir(os.path.dirname(args.output_path))

        with open(args.output_path, 'w') as f:
            tsvout =  csv.writer(f)
            for user in usernames:
                sentences, labels, comments = database.get_user_data(user)
                for i, sentence in enumerate(sentences):
                    label_argmax = np.argmax(labels[i])
                    if label_argmax == 0:
                        current_type = 'A is-a B'
                    elif label_argmax == 1:
                        current_type = 'B is-a A'
                    elif label_argmax == 2:
                        current_type = 'Neither'
                    elif label_argmax == 3:
                        current_type = 'Skip'
                    else:
                        raise AssertionError("label length should be 4. It's now %s" %(str(labels[i])))
                    tsvout.writerow([user, ' '.join(sentence), current_type, comments[i]])

