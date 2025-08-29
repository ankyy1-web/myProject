from main import get_chroma_collection
coll = get_chroma_collection()
print("Total documents in Chroma:", coll.count())
