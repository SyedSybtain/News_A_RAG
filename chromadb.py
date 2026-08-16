from chromadb import Client

client = Client()
collection = client.create_collection(name="test")
collection.add(
    documents=["Banana", "Pakistan", "Obstinate"],
    ids=["1", "2","3"]
)

results = collection.query(
    query_texts=["Stubborn"],
    n_results=3
)   

print("IDs : ", results['ids'])
print("Distance : ",results['distances'])
