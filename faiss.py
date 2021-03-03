import faiss

dimension = 3

v_1 = np.array([[1,2,3]]).astype('float32')
v_2 = np.array([[4,5,6]]).astype('float32')
v_3 = np.array([[7,8,9]]).astype('float32')

index = faiss.IndexIDMap2(faiss.IndexFlat(dimension))

index.add_with_ids(v_1, np.array([10]))
index.add_with_ids(v_2, np.array([20]))
index.add_with_ids(v_3, np.array([30]))


print("ntotal")
print(index.ntotal)
print()

print("Reconstruct")
v_test = index.reconstruct(30)
print(v_test)
print()

print("Searching")
D, I = index.search(v_1, 10)
print(D)
print()
print(I)
print()

faiss.write_index(index, "./index")
