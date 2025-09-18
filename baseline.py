from models.res import Resnet50FeatureExtractor, CustomImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import json
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import matplotlib.patches as patches

def image_feature_extract(
    image_height,
    image_width,
    batch_size,
    device,
    image_dir
):
    image_transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0]
        ),
    ])

    custom_image_dataset = CustomImageDataset(
        image_dir,
        transform=image_transform,
    )

    image_loader = DataLoader(
        custom_image_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    fx = Resnet50FeatureExtractor(device=device)

    image_path_list = []
    image_feature_list = []

    for batch_idx, (data) in enumerate(image_loader):
        image_path = data['path']
        image_path_list.append(image_path)

        img = data['image']
        extracted_feature = fx(img.to(device=device))

        # print(image_path)
        for i in extracted_feature.keys():
            img_feature = extracted_feature[i].flatten().cpu().detach().numpy()
            image_feature_list.append(img_feature)
            # print(img_feature.shape)

    print(len(image_feature_list))
    print('Done feature extraction.')
    print(image_path_list)

    return (image_feature_list, image_path_list)

def visualize_clusters(image_feature_embedding, kmeans_labels, image_paths, cluster_centers):
    plt.figure(figsize=(10, 10))

    # create a color map for clusters
    num_clusters = len(set(kmeans_labels))
    cmap_name = 'hsv'

    # Plot images with clusters
    for label in set(kmeans_labels):
        label_indices = np.where(kmeans_labels == label)[0]
        color = plt.cm.get_cmap(cmap_name)(label / num_clusters) # get the color based on the cluter labels

        for idx in label_indices:
            # scatter diagram
            # plt.scatter(image_feature_embedding[idx, 0], image_feature_embedding[idx, 1], c='b', alpha=0.5)  

            # add image
            img = plt.imread(image_paths[idx][0])

            x = image_feature_embedding[idx][0]
            y = image_feature_embedding[idx][1]

            plt.imshow(img, extent=(x, x + img.shape[1] * 0.08, 
                                    y, y + img.shape[0] * 0.08),
                                    aspect='auto', alpha=1.0, cmap=cmap_name)

            # create a rectangle patch to fill the color
            plt.fill_between([x, x + img.shape[1] * 0.08], y, y + img.shape[0] * 0.08, color=color, alpha = 0.1)
    
    # Plot cluster centers         
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', c='r', s=100)

    # draw outer circle
    for center in cluster_centers:
        # find the nearest neighbors
        neigh = NearestNeighbors(radius=200.0)  
        neigh.fit(image_feature_embedding)
        indices = neigh.radius_neighbors([center], 0.1, return_distance=False)[0]

        # check if the sample set is empty
        if len(indices) == 0: continue
        else: print(indices)

        # calculate the outer circle
        cluster_samples = image_feature_embedding[indices]
        circle_center = cluster_samples.mean(axis=0)    
        circle_radius = max(distance.euclidean(circle_center, sample) for sample in cluster_samples)

        # plot circle
        circle = patches.Circle(circle_center, circle_radius, edgecolor = 'r', facecolor='none', linewidth=2)
        plt.gca().add_patch(circle)

    plt.title('Cluster Visualization')
    plt.show()

def cluster_images(image_feature_list, image_path_list):
    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=18)
    img_feature_embedding = pca.fit_transform(image_feature_list)

    silhouette_scores = []
    max_clusters = 10

    for num_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.array(img_feature_embedding))

        # compute the silhouette score
        silhouette_avg = silhouette_score(np.array(img_feature_embedding), kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    # find the optimal number of clusters
    optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"Optimal number of clusters: {optimal_num_clusters}")

    # use the optimal number of clusters for final clustering
    # optimal_num_clusters = 7
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=0).fit(np.array(img_feature_embedding))

    # visualize the clustering results in the 2D map
    visualize_clusters(img_feature_embedding, kmeans.labels_, image_path_list, kmeans.cluster_centers_)

    # Store images paths in clusters
    image_cluster_dict = {}
    for i, m in enumerate(kmeans.labels_):
        if str(m) not in image_cluster_dict:
            image_cluster_dict[str(m)] = []
        image_cluster_dict[str(m)].append(image_path_list[i])

    # Save clustered images to corresponding directories
    save_path = 'E:/jade-typological-judge/jade_clusters'    # change it to your own path
    os.makedirs(save_path, exist_ok=True)
    for cluster_label, cluster_images in image_cluster_dict.items():
        cluster_dir = os.path.join(save_path, f"cluster_{cluster_label}")
        os.makedirs(cluster_dir, exist_ok=True)
        for img_path in cluster_images:
            img = Image.open(img_path[0])
            img.save(os.path.join(cluster_dir, os.path.basename(img_path[0])))

    # Print and save cluster information
    print(json.dumps(image_cluster_dict, indent=4, separators=(',', ':')))
    with open(os.path.join(save_path, 'cluster_info.json'), 'w') as f:
        json.dump(image_cluster_dict, f, indent=4, separators=(',', ':'))

    # Visualize some sample images from each cluster
    # f, axarr = plt.subplots(number_clusters, 5)
    # for i, (cluster_label, cluster_images) in enumerate(image_cluster_dict.items()):
    #     for j, img_path in enumerate(cluster_images[:5]):
    #         img = Image.open(img_path[0])
    #         axarr[i, j].imshow(np.array(img))
    #         axarr[i, j].set_title(f'Cluster {cluster_label}')
    # plt.show()

    print('Done clustering.')

if __name__ == '__main__':
    image_height = 512
    image_width = 512
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_dir = r'jade'

    cluster_images(
        *image_feature_extract(
            image_height=image_height,
            image_width=image_width,
            batch_size=batch_size,
            device=device,
            image_dir=image_dir
        )
    )