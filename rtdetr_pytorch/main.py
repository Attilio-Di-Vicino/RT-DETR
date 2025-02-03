import argparse
from dataset2.downloader import download_all_images

def main():
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      'image_list',
      type=str,
      default=None,
      help=('Filename that contains the split + image IDs of the images to '
            'download. Check the document'))
  parser.add_argument(
      '--num_processes',
      type=int,
      default=5,
      help='Number of parallel processes to use (default is 5).')
  parser.add_argument(
      '--download_folder',
      type=str,
      default=None,
      help='Folder where to download the images.')
  download_all_images(vars(parser.parse_args()))

if __name__ == "__main__":
    main()