import os, faiss
import numpy as np


class SimilarityIndex:
    def __init__(self, index_path: str, index_dimension: int):

        self.index_path = index_path

        # Load index if it exists
        if os.path.isfile(index_path):
            self.index = faiss.read_index(index_path)

        # Create index if it does not exist
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlat(index_dimension, faiss.METRIC_INNER_PRODUCT))
            faiss.write_index(self.index, self.index_path)

    def add_vector_to_index(self, vector: list, vector_id: int):
        faiss.normalize_L2(vector)
        self.index.add_with_ids(vector, np.array([vector_id], dtype=np.int64))
        faiss.write_index(self.index, self.index_path)
        return vector_id

    def get_similar_vectors(self, query_vector_id: int, amount=10, threshold: float = 0) -> dict:
        self.index = faiss.read_index(self.index_path)
        query_vector = np.array([self.index.reconstruct(query_vector_id)])
        total_vectors_in_index = self.index.ntotal

        if amount >= total_vectors_in_index:
            amount = total_vectors_in_index - 1

        # index search
        distances, indexes = self.index.search(
            query_vector, amount + 1
        )  # +1 to compensate for query vector always beeing the the nearest vector

        sim_cases = {}
        for index_id, similarity in zip(indexes[0], distances[0]):
            if index_id != query_vector_id and similarity >= threshold:
                sim_cases[int(index_id)] = float(similarity)

        return sim_cases

    def reconstruct_vector(self, query_vector_id: int):
        self.index = faiss.read_index(self.index_path)
        try:
            reconstructed_vector = np.array([self.index.reconstruct(query_vector_id)])
        except KeyError or ValueError:
            reconstructed_vector = np.array([])

        return reconstructed_vector

    def get_number_of_vectors_in_index(self):
        self.index = faiss.read_index(self.index_path)
        return self.index.ntotal

    def delete_vector_from_index(self, vector_id: int):
        self.index = faiss.read_index(self.index_path)
        self.index.remove_ids(np.array([vector_id]))
        faiss.write_index(self.index, self.index_path)
        return vector_id
