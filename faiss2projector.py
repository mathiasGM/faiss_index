# Helper functions for faiss2projector. 
# Concept: Take a Faiss index, retrieve its vectors and visualize them using tensorflow embedding projector

import os, faiss
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector


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


def write_meta_data(index_dict, log_dir):

    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for id_ in index_dict.values():
            f.write("{}\n".format(id_))


def set_checkpoint_and_config(embeddings, log_dir):
    # Save embeddings to tensorflow board
    embeddings = tf.Variable(embeddings)

    checkpoint = tf.train.Checkpoint(embedding=embeddings)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()

    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)

# Run: 
#tensorboard --logdir logs
