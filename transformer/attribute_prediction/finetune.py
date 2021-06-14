import argparse
import time
import tqdm
import shutil
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import attribute_loader
from efficientnet_pytorch import EfficientNet


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    return


def compute_metric(pred, label, topk=5):
    topk_tags = pred.topk(dim=1, k=topk, largest=True)[1]
    # compute precision at topk
    positive = label.gather(dim=1, index=topk_tags)
    p = positive.sum(dim=1) / topk
    r = positive.sum(dim=1) / (label.sum(dim=1)+1e-5)
    score = 2 * p * r / (p + r + 1e-5)
    return score.sum() / label.size(0)


def evaluate_model(data_loader, model, loss, device):
    model.eval()
    test_num = 0
    # Update learning rate and create optimizer
    error_sum = 0.0
    # Training loop
    fs_sum = 0.0
    for i, (images, labels, _) in enumerate(data_loader):
        # Set mini-batch dataset
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            outs = model(images)
            error = loss(outs, labels)
            fs = compute_metric(outs, labels)

            error_sum += error.item() * images.size(0)
            fs_sum += fs.item() * images.size(0)

            test_num += images.size(0)

    return error_sum / test_num, fs_sum / test_num


def finetune_attributes(args):
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    # Build data loader
    # --Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    transform_dev = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image_folder = args.image_folder.format(args.data_set)
    data_file = args.data_file
    train_data_loader = attribute_loader.get_loader(
        root=image_folder,
        data_file=data_file.format(args.data_set, 'train'),
        class_file=args.label_file,
        transform=transform,
        batch_size=args.batch_size,
        shuffle=True)

    val_data_loader = attribute_loader.get_loader(
        root=image_folder,
        data_file=data_file.format(args.data_set, 'val'),
        class_file=args.label_file,
        transform=transform_dev,
        batch_size=args.batch_size,
        shuffle=False)

    # Load the models
    model = EfficientNet.from_pretrained('efficientnet-b7')
    model_type = 'ft'
    ckpt = torch.load(args.pretrained_model, map_location='cpu')
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)

    # - freeze the bottom part
    trainable_parameters = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            param.requires_grad = True
            trainable_parameters.append(param)
        else:
            param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    model.to(device)

    # Loss and optimizer
    current_lr = args.learning_rate
    optimizer = torch.optim.Adam(lr=current_lr, params=trainable_parameters)
    bce_average = nn.BCEWithLogitsLoss(reduction='mean').to(device)

    # Experiment logging
    global_step = 0
    cur_patient = 0
    best_score = float('-inf')
    total_step = len(train_data_loader)

    save_folder = 'logs/{}-{}'.format(
        args.data_set, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(save_folder, scripts_to_save=[])

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(save_folder, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    for epoch in range(1000):
        if global_step % args.checkpoint == 0:
            model.eval()
            logging('-' * 87)
            with torch.no_grad():
                error, fs = evaluate_model(
                    val_data_loader, model, bce_average, device)
            logging(
                '| ({}) eval loss: {:8.3f} | score {:8.5f} / {:8.5f}'.format(
                    epoch, error, fs, best_score))
            logging('-' * 87)
            # print(metrics)
            dev_score = fs
            if dev_score > best_score:
                best_score = dev_score

                torch.save(model.state_dict(), os.path.join(
                    args.save_model.format(args.data_set, model_type)))
            else:
                cur_patient += 1
                if cur_patient >= args.patient:
                    current_lr /= 2
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                    if current_lr < args.learning_rate * 1e-3:
                        # stop_train = True
                        break
                    cur_patient = 0

        model.train()
        for images, labels, _ in tqdm.tqdm(
                train_data_loader,
                desc="training epoch {}".format(epoch)):
            # Set mini-batch dataset
            images = images.to(device)
            labels = labels.to(device)
            outs = model(images)
            error = bce_average(outs, labels)

            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            global_step += 1
            # Print log info
            if global_step % args.log_step == 0:
                if global_step >= total_step:
                    global_step = (global_step - int(global_step / total_step) *
                                   total_step)

                logging('| epoch {:3d} | step {:6d}/{:6d} | '
                        'lr {:06.6f} | train loss {:8.3f}'.format(
                            epoch, global_step, total_step, current_lr,
                            error.item()))

    logging('beset_dev_score: {}'.format(best_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--image_folder', type=str,
                        default='../resized_images/{}/')
    parser.add_argument('--data_file', type=str,
                        default='data/asin2attr.{}.{}.json')
    parser.add_argument('--label_file', type=str,
                        default='data/attribute2idx.json')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--data_set', type=str, default='dress',
                        help='dress / shirt / toptee')

    # model
    parser.add_argument('--pretrained_model', type=str,
                        default='deepfashion_models/'
                                'dfattributes_efficientnet_b7ns.pth')
    parser.add_argument('--save_model', type=str,
                        default='models/attributes_{}_{}.pth',
                        help='path for saving trained models')
    parser.add_argument('--loss', type=str, default='binary',
                        help='binary / rank')

    parser.add_argument('--log_step', type=int, default=45,
                        help='step size for printing log info')
    parser.add_argument('--checkpoint', type=int, default=1,
                        help='step size for saving models')
    parser.add_argument('--patient', type=int, default=3,
                        help='patient for reducing learning rate')

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    print(args)
    finetune_attributes(args)
