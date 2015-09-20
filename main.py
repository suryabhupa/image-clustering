from __future__ import division 
from sklearn.cluster import KMeans
import dropbox
from PIL import Image
import numpy as np
from clarifai.client import ClarifaiApi
import os

surya_access_token = "Y-HjafV0lKEAAAAAAAAlcFwacpgYDO_Ouf_KZ0SrHFZTYPqa5eK1kZvW2KaQ0fOw" 
client = dropbox.client.DropboxClient(surya_access_token)
images_folder = "/sample_photos_copy"

def get_images(client, images_folder):
    """
    Given an images-folder name returns all the images in <128x128?> format. 
    
    Params:
        client: object, owner of the dropbox acc we are hacking.
        images_folder: string, name of the folder containing the images 
                       (must be in home page, for now)
    Return:
        images: list containing JPEG images in the desired format.
        
        http://stackoverflow.com/questions/12798885/python-image-library-convert-pixels-to-2d-list
        http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """
    metadata = client.metadata(images_folder)
    vectorized_images = []
    images_data = metadata["contents"] #list, each item contains data of an image. "path" contains the image path/name
    print len(images_data)
    for image in images_data:
        print len(image)
        file_name = image["path"] 
        data = client.get_file(image["path"]).read() 
        with open("output.png", 'w+') as f:
            f.write(data) 
            embedding = clarifai_api.embed(f)['results'][0]['result']['embed']
        vectorized_images.append({'id': file_name, 'data': embedding})

    return vectorized_images

def k_means(vectorized_images, k):
    """
    Given vectorized images and cluster amount k, performs k-means on the images. Returns a matrix with the centroids and an array
    with cluster assignments
    
    Params:
        vectorized_images: list of dictionaries with 'id' and 'data' keys specifying 
                           names and vector representations of the images
        k: amount of clusters
    Return:
        centroids: k points representing the centers of the clusters
        cluster_assignments: an array of size n, entry i is the cluster assignment of the ith image
        sum_distances: sum of distances from each element to its center, helps us know how good the k-means did.
    """
    data = map(lambda image_dic: image_dic["data"], vectorized_images)
    alg = KMeans(n_clusters=k)
    alg.fit(data)
    centroids = alg.cluster_centers_
    cluster_assignments = alg.labels_ 
    sum_distances = alg.inertia_ 
    return centroids, cluster_assignments, sum_distances
    
def cluster_up_images (vectorized_images, cluster_assignments, k):
    """
    Given images and their corresponding assignments, returns the images grouped by cluster.
    
    Params:
        vectorized_images: images in the format returned by images_to_vectors.
        k: amount of clusters
        cluster_assignments: an array of size n, entry i is the cluster assignment of the ith image
    Return:
        clustered_images: list of lists where each sublist contains all the images belonging to the same cluster
    """
    #k = max(cluster_assignments)+1
    clustered_images = [[] for x in range(k)]
    for index in range(len(vectorized_images)):
        image = vectorized_images[index]
        image_assignment = cluster_assignments[index]
        clustered_images[image_assignment].append(image)
    return clustered_images

def get_top_tags(folder_path):
    '''
    Param:
            folder_path: path to the folder containing images
    Return:
            top_tags: list of top three tags for all of the images in a folder using Clarfai API

    '''
    top_tags = collections.defaultdict(float)
    for (_, _, filenames) in walk(folder_path):
        for filename in filenames:
            try:
                result = clarifai_api.tag_images(open(folder_path + "/" + filename))
                top_ten_tag = result['results'][0]['result']['tag']['classes'][:10]
                top_ten_prob = result['results'][0]['result']['tag']['probs'][:10]
                for tag,prob in zip(top_ten_tag, top_ten_prob): 
                    top_tags[tag] += prob
            except: 
                pass
    sorted_tags = sorted(top_tags.items(), key=operator.itemgetter(1))[::-1]
    return [tag[0] for tag in sorted_tags[:3]]

# for i, image in enumerate(get_images(client, "sample_photos")):
#        client.put_file("/sample_photos/"+str(i), image.read())

def insert_clustered_images(clustered_images):
    for index in range(len(clustered_images)):
        images = clustered_images[index]
        client.file_create_folder(images_folder + "/" + str(index))
        for image in images:
            client.file_move(image["id"], images_folder + "/" + str(index))
    

print "Welcome to Mirage!"
clarifai_api = ClarifaiApi() 
vectorized_images = get_images(client, images_folder)
c, ca, sd = k_means(vectorized_images, 5)
print ca
clustered_images = cluster_up_images(vectorized_images, ca, 5) 
# insert_clustered_images(clustered_images)
