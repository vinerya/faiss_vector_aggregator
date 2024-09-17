import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document

def create_new_faiss_index(representative_embeddings, metadata_by_column, normalize_L2=False):
    embedding_dimension = list(representative_embeddings.values())[0].shape[0]
    new_index = faiss.IndexFlatL2(embedding_dimension)

    embeddings_list = []
    new_docstore = InMemoryDocstore({})
    new_index_to_docstore_id = {}

    for idx, (column_value, embedding) in enumerate(representative_embeddings.items()):
        if normalize_L2:
            embedding = embedding / np.linalg.norm(embedding)
        embeddings_list.append(embedding)
        
        # Get the metadata from metadata_by_column
        meta = metadata_by_column.get(column_value, {'id': column_value})
        
        # Assign new document ID for the new index
        new_doc_id = f"doc_{idx}"
        new_index_to_docstore_id[idx] = new_doc_id
        doc = Document(page_content="", metadata=meta)
        new_docstore._dict[new_doc_id] = doc

    embeddings_array = np.vstack(embeddings_list).astype('float32')
    new_index.add(embeddings_array)

    # Save only docstore and index_to_docstore_id to match expected format
    new_metadata = (new_docstore, new_index_to_docstore_id)
    return new_index, new_metadata
