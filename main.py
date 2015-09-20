from __future__ import division
from sklearn.cluster import KMeans
from clarifai.client import ClarifaiApi
from PIL import Image
import numpy as np
import os, collections, operator, dropbox

surya_access_token = "Y-HjafV0lKEAAAAAAAAlcFwacpgYDO_Ouf_KZ0SrHFZTYPqa5eK1kZvW2KaQ0fOw"
client = dropbox.client.DropboxClient(surya_access_token)
images_folder = "/sample_photos_copy"

def get_images(client, images_folder):
    metadata = client.metadata(images_folder)
    vectorized_images = []
    images_data = metadata["contents"]
    for image in images_data:
        file_name = image["path"]
        print image["path"]
        data = client.get_file(image["path"]).read()

        with open("output.png", 'w+') as f:
            f.write(data)
            embedding = clarifai_api.embed(f)['results'][0]['result']['embed']
        vectorized_images.append({'id': file_name, 'data': embedding})

    return vectorized_images

def k_means(vectorized_images, k):
    data = map(lambda image_dic: image_dic["data"], vectorized_images)
    alg = KMeans(n_clusters=k)
    alg.fit(data)
    centroids = alg.cluster_centers_
    cluster_assignments = alg.labels_
    sum_distances = alg.inertia_
    return centroids, cluster_assignments, sum_distances

def cluster_up_images (vectorized_images, cluster_assignments, k):
    clustered_images = [[] for x in range(k)]
    for index in range(len(vectorized_images)):
        image = vectorized_images[index]
        image_assignment = cluster_assignments[index]
        clustered_images[image_assignment].append(image)
    return clustered_images

def get_top_tags(folder_path):
    metadata = client.metadata(folder_path)
    top_tags = collections.defaultdict(float)
    images_data = metadata["contents"]

    for image in images_data:
        file_name = image["path"]
        data = client.get_file(image["path"]).read()

        try:
            with open("output.png", 'w+') as f:
                f.write(data)
                result = clarifai_api.tag_images(f)
                top_ten_tag = result['results'][0]['result']['tag']['classes'][:10]
                top_ten_prob = result['results'][0]['result']['tag']['probs'][:10]
                for tag,prob in zip(top_ten_tag, top_ten_prob):
                    top_tags[tag] += prob
        except:
            pass
    sorted_tags = sorted(top_tags.items(), key=operator.itemgetter(1))[::-1]
    return [tag[0] for tag in sorted_tags[:3]]

def insert_clustered_images(clustered_images):
    for index in range(len(clustered_images)):
        images = clustered_images[index]
        folder_path = "clusters/" + str(index)
        client.file_create_folder(folder_path)
        for image in images:
            client.file_copy(image["id"], "clusters/" + str(index) + "/" + image["id"][image["id"].rfind('/')+1:])
        print get_top_tags(folder_path)
        client.file_move("clusters/" + str(index), "clusters/" + '_'.join(get_top_tags(folder_path)))

print "Welcome to Mirage!"
clarifai_api = ClarifaiApi()
vectorized_images = get_images(client, images_folder)
c, ca, sd = k_means(vectorized_images, 5)
print ca
clustered_images = cluster_up_images(vectorized_images, ca, 5)
insert_clustered_images(clustered_images)
