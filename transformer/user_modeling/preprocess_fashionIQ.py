import numpy
import json
import argparse
import csv
import os




def preprocess_data(file_prefix, topic):
    """load tsv_file and save image1_path and image2_path and caption in json and save it.

    Args:
    tsv_file_path

    Returns:
    None
    """
    IMAGE_ROOT = 'resized_images/{}/'
    CAPT = 'caption/pair2cap.{}.{}.json'
    LABEL = 'attribute_prediction/attributes/{}_attribute_best.pth_{}_attributes.json'
    # DICT = 'attribute_prediction/attribtue2idx.json'
    # SPLIT = 'image_splits/split.{}.{}.json'

    #training file
    training_json = json.load(open(file_prefix + CAPT.format(topic, 'train')))
    dev_json = json.load(open(file_prefix + CAPT.format(topic, 'val')))
    test_json = json.load(open(file_prefix + CAPT.format(topic, 'test')))
    image_path = file_prefix + IMAGE_ROOT.format(topic)
    data_train = []
    data_dev = []
    data_test = []
    data_dev_combine = []
    data_test_combine = []


    label_path = LABEL.format(topic, topic)
    label_json = json.load(open(label_path))
    label_dic = {}

    for line in label_json:
        label_dic[line['image']] = {}
        label_dic[line['image']]["id"] = line["predict_id"]
        label_dic[line['image']]["pred"] = line["prediction"]

    for entry in training_json:

        image0 = image_path + entry["candidate"] + '.jpg'

        image0_label = label_dic[entry["candidate"]+ '.jpg']["id"]

        image1 = image_path + entry["target"] + ".jpg"

        image1_label = label_dic[entry["target"]+ '.jpg']["id"]

        caps = entry["captions"]

        for cap in caps:

            data_train.append({'image0': image0, 'image1':image1, "captions":cap, \
                                "image0_label":image0_label,
                                "image1_label":image1_label,
                                })

    for entry in dev_json:
        image0 = image_path + entry["candidate"] + '.jpg'
        image0_label = label_dic[entry["candidate"]+ '.jpg']["id"]
        image1 = image_path + entry["target"] + ".jpg"
        image1_label = label_dic[entry["target"]+ '.jpg']["id"]
        caps = entry["captions"]

        for cap in caps:

            data_dev.append({'image0': image0, 'image1':image1, "captions":cap,\
                                "image0_label":image0_label,
                                "image1_label":image1_label,
                                })

        data_dev_combine.append({'image0': image0, 'image1':image1, "captions":caps,\
                                    "image0_label":image0_label,
                                    "image1_label":image1_label,
                                })

    for entry in test_json:
        image0 = image_path + entry["candidate"] + '.jpg'
        image0_label = label_dic[entry["candidate"]+ '.jpg']["id"]
        image1 = image_path + entry["target"] + ".jpg"
        image1_label = label_dic[entry["target"]+ '.jpg']["id"]
        caps = entry["captions"]
      

        for cap in caps:

            data_test.append({'image0': image0, 'image1':image1, "captions":cap, \
                                "image0_label":image0_label,
                                "image1_label":image1_label,
                            })

        data_test_combine.append({'image0': image0, 'image1':image1, "captions":caps, 
                                    "image0_label":image0_label,
                                    "image1_label":image1_label,
                             
                                    })

    file_prefix += "/" + topic

    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)

    with open(file_prefix +  '/data_train.json', 'w') as outfile:
        json.dump(data_train, outfile)

    with open(file_prefix +  '/data_dev.json', 'w') as outfile:
        json.dump(data_dev, outfile)

    with open(file_prefix + '/data_test.json', 'w') as outfile:
        json.dump(data_test, outfile)

    with open(file_prefix + '/data_dev_combine.json', 'w') as outfile:
        json.dump(data_dev_combine, outfile)

    with open(file_prefix + '/data_test_combine.json', 'w') as outfile:
        json.dump(data_test_combine, outfile)

   




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


def preprocess_data_from_xiaoxiao(file_prefix, topic):
    IMAGE_ROOT = 'resized_images/{}/'
    CAPT = 'caption/pair2cap.{}.{}.json'
    LABEL = 'data/fashion-IQ/predicted_attributes/{}_attribute_best.pth_{}_attributes.json'
    DICT = 'attribute_prediction/attribute2idx.json'
    # SPLIT = 'image_splits/split.{}.{}.json'

    training_json = json.load(open(file_prefix + CAPT.format(topic, 'train')))
    dev_json = json.load(open(file_prefix + CAPT.format(topic, 'val')))
    test_json = json.load(open(file_prefix + CAPT.format(topic, 'test')))
    image_path = file_prefix + IMAGE_ROOT.format(topic)
    data_train = []
    data_dev = []
    data_test = []
    data_dev_combine = []
    data_test_combine = []


    label_predict = LABEL.format(topic, topic)
    label_predict = json.load(open(label_predict))


    label_dic = json.load(open(DICT))

    # for line in label_json:
    #     label_dic[line['image']] = {}
    #     label_dic[line['image']]["id"] = line["predict_id"]
    #     label_dic[line['image']]["pred"] = line["prediction"]

    for entry in training_json:

        image0 = image_path + entry["candidate"] + '.jpg'

        image0_label = [ label_dic[x] for x in label_predict[entry["candidate"]]["predict"]]
        image0_full_score = label_predict[entry["candidate"]]["full_predict"]

        image1 = image_path + entry["target"] + ".jpg"

        image1_label = [ label_dic[x] for x in label_predict[entry["target"]]["predict"]]
        image1_full_score = label_predict[entry["target"]]["full_predict"]

        caps = entry["captions"]

        for cap in caps:

            data_train.append({'image0': image0, 'image1':image1, "captions":cap, \
                                "image0_label":image0_label,
                                "image1_label":image1_label,
                                "image0_full_score":image0_full_score,
                                "image1_full_score":image1_full_score
                                })

    for entry in dev_json:
        image0 = image_path + entry["candidate"] + '.jpg'
        image0_label = [ label_dic[x] for x in label_predict[entry["candidate"]]["predict"]]
        image1 = image_path + entry["target"] + ".jpg"
        image1_label = [ label_dic[x] for x in label_predict[entry["target"]]["predict"]]
        caps = entry["captions"]
        image0_full_score = label_predict[entry["candidate"]]["full_predict"]
        image1_full_score = label_predict[entry["target"]]["full_predict"]
        for cap in caps:

            data_dev.append({'image0': image0, 'image1':image1, "captions":cap,\
                                "image0_label":image0_label,
                                "image1_label":image1_label,
                                "image0_full_score":image0_full_score,
                                "image1_full_score":image1_full_score
                                })

        data_dev_combine.append({'image0': image0, 'image1':image1, "captions":caps,\
                                    "image0_label":image0_label,
                                    "image1_label":image1_label,
                                    "image0_full_score":image0_full_score,
                                    "image1_full_score":image1_full_score
                                })

    for entry in test_json:
        image0 = image_path + entry["candidate"] + '.jpg'
        image0_label = [ label_dic[x] for x in label_predict[entry["candidate"]]["predict"]]
        image1 = image_path + entry["target"] + ".jpg"
        image1_label = [ label_dic[x] for x in label_predict[entry["target"]]["predict"]] 
        caps = entry["captions"]
        image0_full_score = label_predict[entry["candidate"]]["full_predict"]
        image1_full_score = label_predict[entry["target"]]["full_predict"]
        

        for cap in caps:

            data_test.append({'image0': image0, 'image1':image1, "captions":cap, \
                                "image0_label":image0_label,
                                "image1_label":image1_label,
                                "image0_full_score":image0_full_score,
                                "image1_full_score":image1_full_score
                            })

        data_test_combine.append({'image0': image0, 'image1':image1, "captions":caps, 
                                    "image0_label":image0_label,
                                    "image1_label":image1_label,
                                    "image0_full_score":image0_full_score,
                                    "image1_full_score":image1_full_score
                             
                                    })

    file_prefix += "/" + topic

    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)

    with open(file_prefix +  '/data_train.json', 'w') as outfile:
        json.dump(data_train, outfile)

    with open(file_prefix +  '/data_dev.json', 'w') as outfile:
        json.dump(data_dev, outfile)

    with open(file_prefix + '/data_test.json', 'w') as outfile:
        json.dump(data_test, outfile)

    with open(file_prefix + '/data_dev_combine.json', 'w') as outfile:
        json.dump(data_dev_combine, outfile)

    with open(file_prefix + '/data_test_combine.json', 'w') as outfile:
        json.dump(data_test_combine, outfile)

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

    

    preprocess_data_from_xiaoxiao(args.data_prefix, 'dress')
    preprocess_data_from_xiaoxiao(args.data_prefix, 'shirt')
    preprocess_data_from_xiaoxiao(args.data_prefix, "toptee")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_prefix', required=True)
   
    args = parser.parse_args()
    main(args)