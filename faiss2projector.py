"""
Take a Faiss index, retrieve its vectors and visualize them using tensorflow embedding projector
"""

import os, faiss
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector


### TO DO ###

# 1) Go to notebook and add dictionary save with title 
# 2) Read index, dictionary
# 3) Make embeddings projector

def load_faiss_index(path):
    if os.path.isfile(path):
        index = faiss.read_index(path)
        return index
    else:
        raise FileNotFoundError


def retrieve_vectors_faiss(index, index_dict):

    reconstructed_vectors = []

    for i in index_dict.keys():
        reconstructed_vectors.append(np.array(index.reconstruct(i)))

    return reconstructed_vectors


def write_meta_data(index_dict, path):

    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for id_ in index_dict.values():
            f.write("{}\n".format(id_))


# Directory log for tensorboard
log_dir = 'logs'

# Index
index = load_faiss_index("index")
print(index.ntotal)

ids = [10, 20, 30]
ids_dict = {
    10 : "Number 10",
    20 : "Number 20",
    30 : "Number 30",
}


# Save embeddings to tensorflow board
embeddings = tf.Variable(reconstructed_vectors)

checkpoint = tf.train.Checkpoint(embedding=embeddings)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

config = projector.ProjectorConfig()
embedding = config.embeddings.add()

embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)

# Run: 
#tensorboard --logdir logs
