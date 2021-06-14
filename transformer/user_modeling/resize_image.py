import argparse
import os
from PIL import Image

from joblib import Parallel, delayed
import multiprocessing

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        print(image)
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))

def resize_image_operator(image_file, output_file, size, i, num_images):
    with open(image_file, 'r+b') as f:
        with Image.open(f) as img:
            img = resize_image(img, size)
            img.save(output_file, img.format)
    if (i + 1) % 100 == 0:
        print("[{}/{}] Resized the images and saved."
              .format(i + 1, num_images))
    return

def resize_images_parallel(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_cores = multiprocessing.cpu_count()
    print('resize on {} CPUs'.format(num_cores))

    images = os.listdir(image_dir)
    num_images = len(images)
    Parallel(n_jobs=num_cores)(
        delayed(resize_image_operator)(
        os.path.join(image_dir, image),
        os.path.join(output_dir, image),
        size,
        i,
        num_images) for i, image in enumerate(images))


# def main():
#     image_dir = '../data/images/'
#     output_dir = '../data/revised_images'
#     image_size = [256, 256]
#     resize_images(image_dir, output_dir, image_size)

# if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--image_dir', type=str, default='../data/images/',
# #                         help='directory for train images')
# #     parser.add_argument('--output_dir', type=str, default='../data/revised_images',
# #                         help='directory for saving resized images')
# #     parser.add_argument('--image_size', type=int, default=256,
# #                         help='size for image after processing')
# #     args = parser.parse_args()
#     main()

def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images_parallel(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/train2014/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./data/resized2014/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)
