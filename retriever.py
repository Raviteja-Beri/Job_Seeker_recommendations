from langchain.retrievers import BM25Retriever

def build_bm25_index(jobs):
    descriptions = []
    metadatas = []

    for job in jobs:
        desc = job.get("description", "")
        skills = job.get("skills", "")
        if not desc.strip():
            desc = f"{job['title']} at {job['company']}"

        descriptions.append(desc + " " + skills)
        metadatas.append({"idx": len(descriptions) - 1})

    bm25 = BM25Retriever.from_texts(descriptions, metadatas=metadatas)

    return bm25, descriptions
