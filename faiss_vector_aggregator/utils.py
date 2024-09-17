import faiss
import pickle

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def save_faiss_index(index, output_path):
    faiss.write_index(index, output_path)

def load_metadata(metadata_path):
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)

def save_metadata(metadata, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(metadata, f)
