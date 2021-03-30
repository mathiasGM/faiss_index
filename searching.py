import faiss
import numpy as np

# Helper function for searching in Faiss, using the index and two dictionaries which map the Vector Id (Int) to a (String) Id. 
def search_faiss(query_doc_id:str, index, index_dict, id_dict, amount=10):
    
    # Get query_doc_index based on query_doc_id
    query_doc_faiss_index = id_dict.get(query_doc_id)
    
    # Reconstruct vector
    query_vector = np.array([index.reconstruct(query_doc_faiss_index)])
    
    # Search
    D, I = index.search(query_vector, amount+1)
    
    # Skip the first since the query doc id will always be the first one returned
    similar_indexes_result = I[0][1:]
    
    # Convert doc_indexes to 
    retrieved_docs = [index_dict.get(index) for index in similar_indexes_result]
    
    return retrieved_docs
