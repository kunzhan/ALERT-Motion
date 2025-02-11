from run_attack_mdm import run_attacks
import argparse
import codecs as cs


def read_ids():
    split_file = "target_model/TMR/nsim_test.txt"# os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=int, default=0)
    parser.add_argument('--v_str', type=str, default="1")
    args = parser.parse_args()

    id_list = read_ids()
    run_attacks(args.i, id_list, args.v_str)