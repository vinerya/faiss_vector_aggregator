import os
from .utils import load_faiss_index, load_metadata, save_faiss_index, save_metadata
from .embedding_methods import calculate_embedding
from .faiss_index_helpers import create_new_faiss_index

def aggregate_embeddings(input_folder, column_name, output_folder, method="average", weights=None, trim_percentage=0.1):
    faiss_index_path, metadata_path = _construct_paths(input_folder)
    index, index_to_docstore_id, docstore, normalize_L2 = _load_data(faiss_index_path, metadata_path)
    
    embeddings_by_column, metadata_by_column = _collect_embeddings_by_column(index, index_to_docstore_id, docstore, column_name)
    
    representative_embeddings = {}
    for column_value, embeddings in embeddings_by_column.items():
        representative_embeddings[column_value] = calculate_embedding(embeddings, method, weights, trim_percentage)
    
    new_index, new_metadata = create_new_faiss_index(representative_embeddings, metadata_by_column, normalize_L2)
    
    _save_new_index_and_metadata(new_index, new_metadata, output_folder)
    
    return os.path.join(output_folder, "index.faiss"), os.path.join(output_folder, "index.pkl")

def _construct_paths(input_folder):
    faiss_index_path = os.path.join(input_folder, "index.faiss")
    metadata_path = os.path.join(input_folder, "index.pkl")
    return faiss_index_path, metadata_path

def _load_data(faiss_index_path, metadata_path):
    index = load_faiss_index(faiss_index_path)
    metadata = load_metadata(metadata_path)
    
    # Handle different structures of metadata
    if isinstance(metadata, tuple):
        if len(metadata) == 3:
            docstore, index_to_docstore_id, normalize_L2 = metadata
        elif len(metadata) == 2:
            docstore, index_to_docstore_id = metadata
            normalize_L2 = False  # Default value
        else:
            raise ValueError(f"Unexpected number of items in metadata: {len(metadata)}")
    else:
        raise TypeError(f"Unexpected type of metadata: {type(metadata)}")
    
    return index, index_to_docstore_id, docstore, normalize_L2

def _collect_embeddings_by_column(index, index_to_docstore_id, docstore, column_name):
    embeddings_by_column = {}
    metadata_by_column = {}
    n = index.ntotal
    for i in range(n):
        docstore_id = index_to_docstore_id.get(i)
        if docstore_id is None:
            continue
        document = docstore.search(docstore_id)
        if document and isinstance(document.metadata, dict):
            column_value = document.metadata.get(column_name)
            if column_value is not None:
                embedding = index.reconstruct(i)
                if column_value not in embeddings_by_column:
                    embeddings_by_column[column_value] = []
                    # Save the metadata (assuming it's the same for all embeddings with the same column_value)
                    metadata_by_column[column_value] = document.metadata
                embeddings_by_column[column_value].append(embedding)
    return embeddings_by_column, metadata_by_column

def _save_new_index_and_metadata(new_index, new_metadata, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    output_index_path = os.path.join(output_folder, "index.faiss")
    output_metadata_path = os.path.join(output_folder, "index.pkl")
    save_faiss_index(new_index, output_index_path)
    save_metadata(new_metadata, output_metadata_path)
