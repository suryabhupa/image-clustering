from __future__ import division 
from sklearn.cluster import KMeans
from clarifai.client import ClarifaiApi
import dropbox
import numpy as np
import os, collections, operator, dropbox, math


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
    """
    metadata = client.metadata(images_folder)
    vectorized_images = []
    images_data = metadata["contents"] #list, each item contains data of an image. "path" contains the image path/name
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
    alg = KMeans(n_clusters=i)
    alg.fit(data)
    centroids = alg.cluster_centers_
    cluster_assignments = alg.labels_ 
    sum_distances = alg.inertia_ 
    clustered_images = cluster_up_images(vectorized_images, cluster_assignments, i)
    #print'dunn index, k,', i, (get_dunn_index(map(lambda cluster: np.asarray([image["data"] for image in cluster]), clustered_images)))
    return centroids, cluster_assignments, sum_distances
    
    
# implementation reference: https://github.com/jqmviegas/jqm_cvi/blob/master/base.py
# def delta(ck, cl):
#     values = np.ones([len(ck), len(cl)])*10000
    
#     for i in range(0, len(ck)):
#         for j in range(0, len(cl)):
#             values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
#     return np.min(values)
    
# def big_delta(ci):
#     values = np.zeros([len(ci), len(ci)])
    
#     for i in range(0, len(ci)):
#         for j in range(0, len(ci)):
#             values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
#     return np.max(values)

# def get_dunn_index(k_list):
#     """ Dunn index [CVI]
    
#     Parameters
#     ----------
#     k_list : list of np.arrays
#         A list containing a numpy array for each cluster |c| = number of clusters
#         c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
#     """
#     deltas = np.ones([len(k_list), len(k_list)])*1000000
#     big_deltas = np.zeros([len(k_list), 1])
#     l_range = list(range(0, len(k_list)))
    
#     for k in l_range:
#         for l in (l_range[0:k]+l_range[k+1:]):
#             deltas[k, l] = delta(k_list[k], k_list[l])
        
#         big_deltas[k] = big_delta(k_list[k])

#     di = np.min(deltas)/np.max(big_deltas)
#     return di

    
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
        client.file_create_folder("clusters/" + str(index))
        for image in images:
            client.file_copy(image["id"], "clusters/" + str(index) + "/" + image["id"][image["id"].rfind('/')+1:])


def cosine_distance(v1,v2):
    "compute cosine distance of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def semisupervised_clustering(dir_of_folders, dir_of_images):
    """
    Performs the semi-supervised learning that adds new photos to existing clusters
    or adds new clusters to account for the new clusters.
    
    Param:
        dir_of_folders: directory that contains the existing folders of clustered photos
        dir_of_images: directory of images that need to be clustered
        
    Return:
        clustered_assignments: dictionary mapping folder names to list of dictionaries repsenting the images in the folder
    """
    
    average_images_reps = []
    metadata_big = client.metadata(average_image_reps)
    folders_data = metadata_big["contents"]
    
    for folder in folders_data:
        cluster_vectorized_images = get_images(client, folder["path"])
        average_vectorized_images = [sum(i)/len(i) for i in zip(*cluster_vectorized_images)]
        average_images_reps.append({'folder_id': folder["path"], 'reps': average_vectorized_images})
    
    vectorized_images = get_images(client, dir_of_images)
    
    # dictionary mapping existing folder names to all images in that folder
    clustered_assignments = collections.defaultdict(list)
    
    for image in vectorized_images:
        image_name = image['id']
        image_rep = image['data']
        image_cluster = None
        max_distance = 1
        for cluster in average_image_reps:
            cluster_name = cluster['folder_id'] 
            cluster_rep = cluster['reps']
            similarity = cosine_distance(image_rep, cluster_rep)
            if similarity >= max_distance:
                image_cluster = cluster_name
                max_distance = similarity
        clustered_assignments[image_cluster].append(image)
    
    return clustered_assignments

def cosine_distance(v1,v2):
    "compute cosine distance of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def semisupervised_clustering(dir_of_folders, dir_of_images):
    average_image_reps = []
    metadata_big = client.metadata(dir_of_folders)
    folders_data = metadata_big["contents"]
    
    for folder in folders_data:
        cluster_vectorized_images = get_images(client, folder["path"])
        all_vectorized_images = [i["data"] for i in cluster_vectorized_images]
        average_vectorized_images = [sum(i)/len(i) for i in zip(*all_vectorized_images)]
        average_image_reps.append({'folder_id': folder["path"], 'reps': average_vectorized_images})
    
    vectorized_images = get_images(client, dir_of_images) 
    clustered_assignments = collections.defaultdict(list)
    
    for image in vectorized_images:
        image_name = image['id']
        image_rep = image['data']
        image_cluster = None
        max_distance = 0
        for cluster in average_image_reps:
            cluster_name = cluster['folder_id'] 
            cluster_rep = cluster['reps']
            similarity = cosine_distance(image_rep, cluster_rep)
            if similarity >= max_distance:
                image_cluster = cluster_name
                max_distance = similarity
            print similarity, cluster_name, image_name
        clustered_assignments[image_cluster].append(image)
    
    return clustered_assignments

def insert_new_images(client, clustered_assignments):
    items = clustered_assignments.keys()
    for item in items:
        for elt in clustered_assignments[item]:
            client.file_copy(elt['id'], item + "/" + elt['id'][elt['id'].rfind("/")+1:])

print "Welcome to Mirage!"
clarifai_api = ClarifaiApi()
# vectorized_images = get_images(client, images_folder)
# c, ca, sd = k_means(vectorized_images, 5)
# print ca
# clustered_images = cluster_up_images(vectorized_images, ca, 5)
# insert_clustered_images(clustered_images)
vec_imgs = semisupervised_clustering("clusters/", "sample_photos_new/") 
for i in vec_imgs.items():
    print i
insert_new_images(client, vec_imgs)

