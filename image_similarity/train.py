from image_similarity.flower_search import FlowerImageSearch
from utils.plot import plot_image_from_path, plot_images_from_index, plot_image_from_index
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from random import randrange


def search_similar_flower(dir, no_show = False):
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    dir = Path(dir)
    search_tfis = TrainFlowerImageSearch()

    if dir.is_dir():
        image_files = [p for p in dir.rglob('*') if p.suffix.lower() in IMAGE_EXTS]
    else:
        image_files = [dir]

    for image_path in image_files:
        print(image_path.name)
        image = Image.open(image_path)
        results = search_tfis.search(image)
        if not no_show:
            plot_image_from_path(image_path)
            plot_images_from_index(search_tfis.data_loader, results)    

    # search_similar_flower(args.image, no_show = args.no_show)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Train Flower Image Search',
        description='Indexes the flower102 and clusters them to enable searches of similar images')
    parser.add_argument('-n', '--no_show', action='store_true')
    
    args = parser.parse_args()

    tfis = FlowerImageSearch(dataset='datasets/flowers102', model_location='models/kmeans_search/', train = True, batch_size=64)
    tfis.train_fit()

    if not args.no_show:
        example_index = randrange(len(tfis.data_loader.dataset))
        similar_images_indices = tfis.get_similar_cluster(example_index)
        plot_image_from_index(tfis.data_loader, example_index)
        plot_images_from_index(tfis.data_loader, similar_images_indices)
        plt.show()