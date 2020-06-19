"""Takes as input two csv files of recodring ids and their scores (output from score_test_captions, see main_decode_test)
and for each recording, select the captions with the smallest score (score=loss here)
"""

from utils import read_csv_prediction_file, load_gt_captions
import pickle
from clotho_dataloader.data_handling.my_clotho_data_loader import get_clotho_loader, create_dictionaries, modify_vocab
from eval_metrics import evaluate_metrics_from_lists
__author__ = "Thomas Pellegrini - 2020"

data_dir = '../clotho-dataset/data'

LETTER_LIST = pickle.load(open(data_dir + "/characters_list.p", "rb"))
LETTER_FREQ = pickle.load(open(data_dir + "/characters_frequencies.p", "rb"))
# ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
#                'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']
WORD_LIST = pickle.load(open(data_dir + "/words_list.p", "rb"))# 4367 word types
WORD_FREQ = pickle.load(open(data_dir + "/words_frequencies.p", "rb"))

# WORD_COUNT_THRESHOLD = 10
WORD_COUNT_THRESHOLD = None
print("\n !!! WORD_COUNT_THRESHOLD = ", WORD_COUNT_THRESHOLD, " !!!\n")

letter2index, index2letter = create_dictionaries(LETTER_LIST)
word2index, index2word = create_dictionaries(WORD_LIST)

if WORD_COUNT_THRESHOLD is not None:
    print("WORD_COUNT_THRESHOLD =", WORD_COUNT_THRESHOLD)
    word2index, index2word, WORD_LIST, mapping_index_dict = modify_vocab(WORD_LIST, WORD_FREQ, WORD_COUNT_THRESHOLD)
else:
    mapping_index_dict = None
print("Vocab:", len(WORD_LIST) )


# fpath1='checkpoints/seq2seq/clotho/best_model/4367_red_2_2__128_64_0.98_False_False_0.0005_1e-06/val_predicted_captions_greedy.csv'
fpath1='checkpoints/seq2seq/clotho/best_model/4367_red_2_2__128_64_0.98_False_False_0.0005_1e-06/val_predicted_captions_beamsearch_lm_0.50_2g.csv'
fpath2='checkpoints/seq2seq/clotho/best_model/4367_red_2_2__128_64_0.98_False_False_0.0005_1e-06/val_predicted_captions_beamsearch_nolm_bs25_alpha_12.csv'

wav_id_list, captions_dict_pred1 = read_csv_prediction_file(fpath1, add_sos_eos=False)
wav_id_list, captions_dict_pred2 = read_csv_prediction_file(fpath2, add_sos_eos=False)

def read_scores_per_utt(fpath):
    wav_id_list = []
    captions_dict_scores = {}
    captions_list_scores = []
    with open(fpath, "rt") as fh:
        for ligne in fh:
            tab = ligne.rstrip().split(",")
            wav_id_list.append(''.join(tab[0:-1]))
            if 'fisherman' in wav_id_list[-1]:
                # print(wav_id_list[-1])
                wav_id_list.pop()
                wav_id_list.append("09-07-14_2338_Foz, fisherman next to the river.wav")
            if 'cricket.real' in wav_id_list[-1]: print(wav_id_list[-1])
            captions_dict_scores[wav_id_list[-1]] = float(tab[-1])
            captions_list_scores.append(float(tab[-1]))

    print("INFO: predicted scores read from file:", fpath)
    return captions_dict_scores, captions_list_scores

# fpath1='checkpoints/seq2seq/clotho/best_model/4367_red_2_2__128_64_0.98_False_False_0.0005_1e-06/val_predicted_captions_greedy_scores.csv'
fpath1='checkpoints/seq2seq/clotho/best_model/4367_red_2_2__128_64_0.98_False_False_0.0005_1e-06/val_predicted_captions_beamsearch_lm_0.50_2g_scores.csv'
fpath2='checkpoints/seq2seq/clotho/best_model/4367_red_2_2__128_64_0.98_False_False_0.0005_1e-06/val_predicted_captions_beamsearch_nolm_bs25_alpha_12_scores.csv'

s1_dict, s1_list = read_scores_per_utt(fpath1)
s2_dict, s2_list = read_scores_per_utt(fpath2)

print(len(s1_list), len(s2_list))
wins1, wins2 = 0, 0
merged_pred = []
for ind, wav_id in enumerate(wav_id_list):
    # s1, s2 = s1_dict[wav_id], s2_dict[wav_id]
    s1, s2 = s1_list[ind], s2_list[ind]
    if s1 <= s2:
        merged_pred.append(captions_dict_pred1[wav_id])
        wins1 += 1
    else:
        merged_pred.append(captions_dict_pred2[wav_id])
        wins2 += 1


print("wins1, wins2", wins1, wins2)
print(merged_pred[:10])

gt_file = "/clotho_captions_evaluation.pkl"

print("GT CAPTION FILE:", data_dir + gt_file)
captions_gt = load_gt_captions(data_dir + gt_file, wav_id_list)
print(captions_gt[0])

metrics = evaluate_metrics_from_lists(merged_pred, captions_gt)

average_metrics = metrics[0]
for m in average_metrics.keys():
    # print("%s\t%.3f" % (m, average_metrics[m]))
    print("%.3f" % (average_metrics[m]))
    # if "SPIDEr" in m: result_fh.write("%.3f\n" % (average_metrics[m]))