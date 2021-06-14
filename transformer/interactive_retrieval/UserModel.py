import torch
from user_model import load_trained_model
import Vocabulary
from Beam import beam_search, greedy_search
import argparse


class UserModel:
    def __init__(self, opt, mode='greedy_search'):
        self.model = load_trained_model(
            opt.user_model_file.format(opt.data_set))
        self.model.eval()
        self.vocab = Vocabulary.Vocabulary()
        vocab_file = opt.user_vocab_file.format(opt.data_set)
        self.vocab.load(vocab_file)
        self.max_seq_len = opt.max_seq_len
        self.opt = opt
        self.decode_mode = mode
        return

    def to(self, device):
        self.model = self.model.to(device)

    def get_max_seq_len(self):
        return self.max_seq_len

    def get_vocab_size(self):
        return len(self.vocab)

    def get_caption(self, target_img, candidate_img,
                    target_attr, candidate_attr, return_cap=False):
        pad_idx = self.vocab('<pad>')
        if self.decode_mode == 'beam_search':
            packed_results = [beam_search(
                candidate_img[i].unsqueeze(dim=0).unsqueeze(dim=0),
                target_img[i].unsqueeze(dim=0).unsqueeze(dim=0),
                self.model, self.opt, self.vocab,
                candidate_attr[i].unsqueeze(dim=0),
                target_attr[i].unsqueeze(dim=0))
                for i in range(target_img.size(0))]

            pad_cap_idx = []
            caps = []
            for cap in packed_results:
                caps.append(cap[1])
                if len(cap[0]) > self.max_seq_len:
                    pad_cap_idx.append(cap[0][:self.max_seq_len])
                else:
                    pad_cap_idx.append(
                        cap[0] + [pad_idx] * (self.max_seq_len - len(cap[0])))

            pad_cap_idx = torch.tensor(pad_cap_idx, dtype=torch.long)
        else:
            pad_cap_idx, caps = greedy_search(candidate_img.unsqueeze(dim=1),
                                              target_img.unsqueeze(dim=1),
                                              self.model, self.opt, self.vocab,
                                              candidate_attr, target_attr)
        if return_cap:
            return pad_cap_idx, caps
        return pad_cap_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str,
                        default='../resized_images/{}/')

    parser.add_argument('--pretrained_image_model', type=str,
                        default='../attribute_prediction/deepfashion_models/'
                                'dfattributes_efficientnet_b7ns.pth')

    parser.add_argument('--save', type=str, default='../models/ranker/',
                        help='path for saving ranker models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--data_set', type=str, default='dress',
                        help='dress / toptee / shirt')

    parser.add_argument('--rep_type', type=str, default='image',
                        help='all / side_info / image ')

    parser.add_argument('--merger_type', type=str, default='attention',
                        help='attention / sum-image / sum-all / sum-other')

    parser.add_argument('--log_step', type=int, default=44,
                        help='step size for printing log info')
    parser.add_argument('--checkpoint', type=int, default=2,
                        help='step size for saving models')
    parser.add_argument('--patient', type=int, default=3,
                        help='patient for reducing learning rate')

    # User model parameters
    parser.add_argument('--user_model_file', type=str,
                        default='../user_modeling/models/'
                                'dress-efficientnet-b7.chkpt')
    parser.add_argument('--user_vocab_file', type=str,
                        default='../user_modeling/vocab.json')
    parser.add_argument('--max_seq_len', type=int, default=10,
                        help='maximum caption length')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='beam search branch size')

    # Model parameters
    parser.add_argument('--history_input_size', type=int, default=256)
    parser.add_argument('--image_embed_size', type=int, default=256)
    parser.add_argument('--text_embed_size', type=int, default=256)
    parser.add_argument('--vocab_embed_size', type=int, default=256)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_dialog_turns', type=int, default=5)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--clip', type=float, default=10)

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0003)

    args = parser.parse_args()

    model = UserModel(args)



