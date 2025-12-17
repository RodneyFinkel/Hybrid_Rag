from alpha_DocumentContextManager import DocumentContextManager
import logging
logging.basicConfig(level=logging.INFO)

context = DocumentContextManager()

query = "What is machine learning"

print('First call...')
results1 = context.get_similar_documents(query, top_k=5)
print(f"Results: {len(results1)}")

print("Second identical call...")
results2 = context.get_similar_documents(query, top_k=5)
print(f"Got {len(results2)} results")

