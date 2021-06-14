import numpy
import json
import argparse
import csv
def preprocess_data(tsv_file_path):
    """load tsv_file and save image1_path and image2_path and caption in json and save it.

    Args:
    tsv_file_path

    Returns:
    None
    """
    with open(tsv_file_path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        # i = 0
        data_train = []
        data_dev = []
        data_test = []
        for i, row in enumerate(reader):
            if i == 0:
                continue
            image0 = process_url(row[1])
            image1 = process_url(row[5])
            text = row[10]

            if row[8] == 'train':
                data_train.append({'image0': image0, 'image1':image1, "captions":text})
            elif row[8] == 'val':
                data_dev.append({'image0': image0, 'image1':image1, "captions":text})
            elif row[8] == 'test':
                data_test.append({'image0': image0, 'image1':image1, "captions":text})

        data_dev_combine = {}
        data_test_combine = {}

        for data in data_dev:
            key = data["image0"] + data["image1"]
            cap = data["captions"]
            if key in data_dev_combine:
                temp = data_dev_combine[key]
                temp["captions"].append(cap)
            else:
                data_dev_combine[key] = {'image0': data["image0"], 'image1':data["image1"], "captions":[cap]}

        temp = data_dev_combine.values()

        data_dev_new = []

        for t in temp:
            data_dev_new.append(t)
            assert len(t['captions']) > 1

        for data in data_test:
            key = data["image0"] + data["image1"]
            cap = data["captions"]
            if key in data_test_combine:
                temp = data_test_combine[key]
                temp["captions"].append(cap)
            else:
                data_test_combine[key] = {'image0': data["image0"], 'image1':data["image1"], "captions":[cap]}

        temp = data_test_combine.values()

        data_test_new = []

        for t in temp:
            data_test_new.append(t)
            assert len(t['captions']) > 1



    with open('./data_train.json', 'w') as outfile:
        json.dump(data_train, outfile)

    with open('./data_dev.json', 'w') as outfile:
        json.dump(data_dev, outfile)

    with open('./data_test.json', 'w') as outfile:
        json.dump(data_test, outfile)

    with open('./data_dev_combine.json', 'w') as outfile:
        json.dump(data_dev_new, outfile)

    with open('./data_test_combine.json', 'w') as outfile:
        json.dump(data_test_new, outfile)

   




def parse_url(url):
    # print('url', url)
    tokens = url.split('/')
    # print(tokens)
    folder = tokens[4]
    tokens = tokens[5].split('?')
    tokens.reverse()
    file = '.'.join(tokens)
    # print(tokens[1])
    # print(tokens)
    # if len(tokens) > 1:
    #     file = tokens[1]
    # else:
    #     file = 'null'
    # print(tokens[4], tokens[5])
    # print(folder, file)
    return '/dccstor/extrastore/Neural-Naturalist/data/resized_images/' + folder + '.' + file


def process_url(url):
    file = parse_url(url)
    if file[-1] == '.':
        file = file + 'jpg'
    # make_folder(folder)

    # if not os.path.isfile(file):
    #     with open(file, 'wb') as f:
    #         resp = requests.get(url, verify=False)
    #         f.write(resp.content)
    #         f.close()
    return file

def main(args):
    ''' Main function '''

    

    preprocess_data(args.tsv_file_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-tsv_file_path', required=True)
    # parser.add_argument('-train_tgt', required=True)
    # parser.add_argument('-valid_src', required=True)
    # parser.add_argument('-valid_tgt', required=True)
    # parser.add_argument('-save_data', required=True)
    # parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    # parser.add_argument('-min_word_count', type=int, default=5)
    # parser.add_argument('-keep_case', action='store_true')
    # parser.add_argument('-share_vocab', action='store_true')
    # parser.add_argument('-vocab', default=None)
    args = parser.parse_args()
    main(args)
