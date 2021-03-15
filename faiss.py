import faiss, uuid
import numpy as np

dimension = 3

v_1 = np.array([[1,2,3]]).astype('float32')
v_2 = np.array([[4,5,6]]).astype('float32')
v_3 = np.array([[7,8,9]]).astype('float32')

index = faiss.IndexIDMap2(faiss.IndexFlat(dimension))

index.add_with_ids(v_1, np.array([10]))
index.add_with_ids(v_2, np.array([20]))
index.add_with_ids(v_3, np.array([30]))

# Alternative to hardcoded Ids:
# faiss_index_id_random = uuid.uuid1().int>>64
# index.add_with_ids(vector, np.array([faiss_index_id_random],dtype=np.int64))

print("ntotal")
print(index.ntotal)
print()

print("Reconstruct")
v_test = index.reconstruct(30)
print(v_test)
print()

print("Searching")
D, I = index.search(v_1, 3)
print("Distances")
print(D[0])
print()
print("Indexes")
print(I[0])
print()


print("Chaning a vector in the index")
# We want to change the vector with Id: 30 in the index to:
v_new = np.array([[70,80,90]]).astype('float32')

# Step 1) Check if the vector exists in the index
try: 
    v_test = index.reconstruct(30)

    # Step 2) Delete vector with Id: 30
    index.remove_ids(np.array([30]))
    
    # We can check that the vector has been removed
    print("ntotal after deleting vector")
    print(index.ntotal)
    print()

    # Step 3) Add new vector and assign to the same Id: 30
    index.add_with_ids(v_new, np.array([30]))

# RuntimeError is thrown if the vector does not exist in the index
except RuntimeError:
    print("Vector with Id: " + str(30) + " does not exist")
    pass

# We can check that the vector has been removed
print("ntotal after adding the new vector")
print(index.ntotal)
print()    

# We can search again, and vector with Id: 30 should now be significant farther away
print("Search after changing vector")
D, I = index.search(v_1, 3)
print("Distances")
print(D)
print()
print("Indexes")
print(I)
print()


# To search in the index based on an averaged representation of two or more vectors (embeddings): 

to_reconstruct_ids = [10,30] # In this example we want to get a representation based on vector_with_id 10 and 30.
reconstructed_vectors = []

# Reconstruct the vectors
for vector_id in to_reconstruct_ids:
    reconstructed_vectors.append(index.reconstruct(vector_id))

average_input_vector = np.array([np.average(reconstructed_vectors, axis=0)])

print("Searching with average input vector representation")
D, I = index.search(average_input_vector, 3)
print("Distances")
print(D[0])
print()
print("Indexes")
print(I[0])
print()


faiss.write_index(index, "./index")
