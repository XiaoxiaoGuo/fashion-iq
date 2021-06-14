import csv
import ssl
import urllib.request
import os
import requests
import tqdm

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

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
    return 'images/' + folder + '.' + file


def make_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    return

def process_url(url):
    file = parse_url(url).lower()
    if file[-1] == '.':
        file = file + 'jpg'

    # make_folder(folder)

    if not os.path.isfile(file):
        with open(file, 'wb') as f:
            resp = requests.get(url, verify=False)
            f.write(resp.content)
            f.close()

with open('birds-to-words-v1.0.tsv') as tsvfile:
  reader = csv.reader(tsvfile, delimiter='\t')
  # i = 0
  data = []
  for i, row in enumerate(reader):
    # i += 1
    if i == 0:
        continue
    process_url(row[1])
    process_url(row[5])
    # print(i)
    # if i == 10:
    #     break
