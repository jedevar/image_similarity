import torch, torchvision
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.models as models
import h5py, pickle, os
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from PIL import Image

class FlowerImageSearch:
    def __init__(self, dataset: str = 'datasets/flowers102', model_location: str = 'models/kmeans_search/', train: bool = False, batch_size: int = 64) -> None:
        self.dataset = dataset
        self.norm_params = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]} 
        self.transforms = T.Compose([
                            T.Lambda(lambda img: img.convert("RGB")), 
                            T.Resize((128, 128)),
                            T.ToTensor(),
                            T.Normalize( mean=self.norm_params['mean'], std=self.norm_params['std'])])
        os.makedirs(os.path.dirname(model_location), exist_ok=True)
        self.index_location = model_location + 'flower_indices.h5'
        self.model_components_location = model_location + 'fit_params.pkl'
        self.data_loader = None
        self.feature_extractor = None
        self.model_num_features = None
        self.feats = None
        self.labels = None
        self.feats_pca_normalized = None
        self.pca = None  # Initialize PCA
        self.kmeans = None  # Initialize KMeans
        self.kmeans_labels = None
        self.kmeans_inv_index = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        if train:
            self.train_fit()
        else:
            if None in (self.pca, self.kmeans, self.kmeans_inv_index):
                self.load_model()
            if self.data_loader is None:
                self.load_dataset()
            if self.feature_extractor is None:
                self.load_feature_extractor()

    def load_dataset(self):
        print(f"Loading dataset into {self.dataset}...")                
        train_dataset = torchvision.datasets.Flowers102(self.dataset, split = 'train', download=True, transform=self.transforms) 
        val_dataset = torchvision.datasets.Flowers102(self.dataset, split = 'val', download=True, transform=self.transforms) 
        test_dataset = torchvision.datasets.Flowers102(self.dataset, split = 'test', download=True, transform=self.transforms) 
        
        output_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

        print(f'Samples in dataset: {len(output_dataset)}')

        self.data_loader = DataLoader(output_dataset, batch_size=self.batch_size, shuffle=False)

    def load_feature_extractor(self):
        print("Loading feature extractor...")
        resnet152_torch = models.resnet152(pretrained=True)
        layers = list(resnet152_torch.children())[:-1]
        self.model_num_features = 2048 # Known from resnet152
        # Load and prepare model for inference
        self.feature_extractor = torch.nn.Sequential(*layers)
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()    

    @torch.no_grad()
    def _forward_pass(self, image: Image = None):
        if image is None:        
            idx = 0
            self.feats = []
            self.labels = []
            for images, labs in tqdm(self.data_loader, total=len(self.data_loader)):
                batch_size = images.size(0)
                images = images.to(self.device)
                feats = self.feature_extractor(images).view(batch_size, -1)
                feats = feats.detach().cpu().numpy().astype(np.float32)
                self.feats[idx:idx+batch_size] = feats
                self.labels[idx:idx+batch_size] = labs.numpy()
                idx += batch_size
        else:
            image = image.to(self.device)
            feats = self.feature_extractor(image).view(1, -1)
            feats = feats.detach().cpu().numpy().astype(np.float32)
            return feats

    def index_dataset(self):
        print("Indexing dataset...")
        N = len(self.data_loader.dataset)
        with h5py.File(self.index_location, 'w') as hf:
            feat_ds = hf.create_dataset('features', shape=(N, self.model_num_features), dtype='float32')
            label_ds = hf.create_dataset('labels', shape=(N,), dtype='int64')
            self._forward_pass()

            feat_ds[:] = self.feats
            label_ds[:] = self.labels

    def fit_pca(self, num_samples: int, dimensions: int):
        print(f"Fitting PCA with {num_samples} samples and {dimensions} dimensions...")
        np.random.seed(7)
        feats_array = np.asarray(self.feats)

        samples_idx = np.random.choice(feats_array.shape[0], size=min(num_samples, feats_array.shape[0]), replace=False)
        samples = feats_array[samples_idx]

        self.pca = PCA(n_components=dimensions, whiten=False)
        self.pca.fit(samples)

        feats_pca = self.pca.transform(feats_array)
        self.feats_pca_normalized = normalize(feats_pca, axis=1, norm='l2').astype(np.float32)
        print(f'PCA feats normalized [{self.feats_pca_normalized.shape}]: {self.feats_pca_normalized[50:55]}')

    def cluster(self, n_clusters: int):
        print(f"Clustering into {n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=7, n_init="auto")
        self.kmeans.fit(self.feats_pca_normalized)
        self.kmeans_labels = self.kmeans.labels_
        
        self.kmeans_inv_index = {} # Create indices
        for i, lab in enumerate(self.kmeans.labels_):
            k = int(lab)
            try:
                self.kmeans_inv_index[k].append(i)
            except KeyError:
                self.kmeans_inv_index[k] = [i]

    def save_model(self):
        print(f"Saving model components to {self.model_components_location}...")
        
        model_data = {"pca": self.pca,
                      "kmeans": self.kmeans,
                      "norm_params": self.norm_params,
                      "kmeans_inv_index": self.kmeans_inv_index,
                      "dataset_config": {"batch_size": self.batch_size,
                                         "norm_params": self.norm_params}}
        with open(self.model_components_location, 'wb') as f:
            pickle.dump(model_data, f)

    def train_fit(self):
        self.load_dataset()
        self.load_feature_extractor()
        self.index_dataset()
        self.fit_pca(1000, 128)
        self.cluster(102)
        self.save_model()

    def get_similar_cluster(self, example_index):
        kmeans_label = self.kmeans_labels[example_index]
        return self.kmeans_inv_index[kmeans_label]
    
    def load_model(self):
        try:
            with open(self.model_components_location, 'rb') as f:
                model_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'No model found in {self.model_components_location}')

        self.pca = model_data["pca"]
        self.kmeans = model_data["kmeans"]
        self.norm_params = model_data["norm_params"]
        self.kmeans_inv_index = model_data["kmeans_inv_index"]
        self.batch_size = model_data["dataset_config"]["batch_size"]
        self.norm_params = model_data["dataset_config"]["norm_params"]
        
    def index_image(self, index: int):
        if self.data_loader is None:
            print('Dataset loaded')
            self.load_dataset()

        N = len(self.data_loader.dataset)
        if index < 0 or index >= N:
            raise IndexError(f"Index {index} out of range [0, {N})")
               
        tensor_img, label = self.data_loader.dataset[index]
        print(f'Label ({label})')

        if isinstance(tensor_img, Image.Image):
            return None, tensor_img.convert("RGB")
        
        if isinstance(tensor_img, torch.Tensor):  # Expect a torch.Tensor (C,H,W) 
            try:
                print("Converting tensor to PIL (server-side)...")
                arr = tensor_img.detach().cpu().numpy()  # shape (C, H, W) expected

                if arr.ndim == 2:
                    arr = np.expand_dims(arr, 0)  # (1,H,W)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = np.repeat(arr, 3, axis=0)  # (3,H,W)

                if not (arr.ndim == 3 and arr.shape[0] == 3):
                    raise RuntimeError(f"Unexpected tensor shape after conversion: {arr.shape}")

                mean = np.array(self.norm_params["mean"]).reshape(3, 1, 1)
                std = np.array(self.norm_params["std"]).reshape(3, 1, 1)
                arr = (arr * std) + mean # denormalize 

                arr = np.clip(arr, 0.0, 1.0) # clip to [0,1], convert to uint8 in HWC order
                arr = (arr * 255.0).round().astype(np.uint8)   # shape still (C,H,W)
                arr = arr.transpose(1, 2, 0)  # to (H,W,C)

                pil_img = Image.fromarray(arr)  # RGB PIL Image
                return None, pil_img

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed reconstruct:({getattr(tensor_img,'shape',None)}): {e}")
            
    def search(self, image, k: int = 10):
        image = self.transforms(image).unsqueeze(0)    # Transform and unsqueeze image
             
        feats = self._forward_pass(image)   # Evaluate and find image cluster
        feats_pca = self.pca.transform(feats)
        feats_pca_normalized = normalize(feats_pca, axis=1, norm='l2').astype(np.float32)
        kmeans_label = int(self.kmeans.predict(feats_pca_normalized)[0])
        
        similar_images = self.kmeans_inv_index[kmeans_label][:k]
        print(f'Images found (first {k} images): {similar_images}')
        return similar_images
