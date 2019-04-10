import argparse
import re

def parse_run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unique_id", required=False, type=str)
    parser.add_argument("--batch_size", required=False, type=int)
    parser.add_argument('--cuda', required=False, type=str)
    # parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument("--init_lr", required=False, type=float)
    parser.add_argument("--max_epochs", required=False, type=int)
    parser.add_argument("--model_name", required=False, type=str)
    parser.add_argument("--result_folder", required=False, type=str)
    parser.add_argument("--data_folder", required=False, type=str)
    parser.add_argument("--weight_decay", required=False, type=float)
    parser.add_argument("--band_pass", required=False, type=str)
    parser.add_argument("--electrodes", required=False, type=str)
    parser.add_argument("--sampling_rate", required=False, type=int)
    parser.add_argument("--n_seconds_test_set", required=False, type=int)
    parser.add_argument("--n_seconds_valid_set", required=False, type=int)
    parser.add_argument("--data", required=False, type=str)

    # data_folder = '/data/schirrmr/fiederer/nicebot/data',
    # batch_size = 64,
    # max_epochs = 200,
    # cuda = True,
    # result_folder = '/data/schirrmr/fiederer/nicebot/results',
    # model_name = 'eegnet',
    # init_lr = 0.001,
    # weight_decay = 0,
    # band_pass_low = None,
    # band_pass_high = None,
    # electrodes = '*',
    # sampling_rate = 256,
    # n_seconds_test_set = 180,
    # n_seconds_valid_set = 180,
    # data = 'onlyRobotData'

    known, unknown = parser.parse_known_args()
    if unknown:
        print("I don't know these run arguments")
        print(unknown)
        exit()

    known_vars = vars(known)
    [known_vars.pop(entry) for entry in list(known_vars.keys()) if known_vars[entry] is None]
    if known_vars["result_folder"] in ["nan"]:
        known_vars["result_folder"] = None

    if known_vars["cuda"] in ["nan", "False", "no"]:
        known_vars["cuda"] = False
    elif known_vars["cuda"] in ["True", "yes"]:
        print('converting True to True')
        known_vars["cuda"] = True
    else:
        print(f"I don't know this cuda argument: {known_vars['cuda']}")
        exit()

    tmp_band_pass = [int(s) for s in re.findall(r'\d+', known_vars["band_pass"])]
    if not tmp_band_pass:
        known_vars["band_pass"] = [None, None]
    else:
        known_vars["band_pass"] = tmp_band_pass

    return known_vars