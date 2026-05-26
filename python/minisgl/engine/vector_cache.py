import torch
import random
import numpy as np
import torch.nn.functional as F
import faiss
import faiss.contrib.torch_utils  # enables zero-copy PyTorch tensors in FAISS

from minisgl.utils import init_logger
from minisgl.engine.skipping_vector_db import SkippingVectorDB

logger = init_logger(__name__)

DEFAULT_K = 5
HIT_RATE_PROBABILITY = 0.3
# 1.5B model
DEFAULT_1_5B_MODEL_DB_PATH = "/home/yff23/data/semantic-layer-skipping/experiments/batch_20260507_154513_Qwen2.5-1.5B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24/db_ivfpq_subsampled_100pct"
# 3B model
DEFAULT_3B_MODEL_DB_PATH = "/home/yff23/data/semantic-layer-skipping/experiments/batch_20260516_232926_Qwen2.5-3B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24-28-32/db_ivfpq_subsampled_100pct"
# 7B model
DEFAULT_7B_MODEL_DB_PATH = "/home/yff23/data/semantic-layer-skipping/experiments/batch_20260514_024813_Qwen2.5-7B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24/db_ivfpq_subsampled_100pct"

DEFAULT_BACKEND = "ivfpq_centroids" # ivfpq, sim, ivfpq_centroids, hot_cache
DEFAULT_METADATA = "ivfpq_store" # ivfpq_store, distribution
DEFAULT_COMPRESSION = "pca" # normal, int8, pca
DEFAULT_NUM_CACHE_VECTORS = 4096
DEFAULT_N_PROBE = 128


# torch.compile generates fused kernels avoiding intermediate VRAM writes
@torch.compile(dynamic=False)
def _fused_pca_search(queries: torch.Tensor, W: torch.Tensor, C_R: torch.Tensor) -> torch.Tensor:
    reduced_queries = torch.matmul(queries, W)
    return torch.matmul(reduced_queries, C_R.T)

@torch.compile(dynamic=False)
def _fused_compressed_search(queries: torch.Tensor, C_int: torch.Tensor) -> torch.Tensor:
    # cast to query dtype and reverse the 127.0 scale factor to restore original float values
    C_float = C_int.to(queries.dtype) / 127.0
    return torch.matmul(queries, C_float.T)


class VectorCache:

    def __init__(self, num_blocks: int, hidden_size: int, device: torch.device, dtype: torch.dtype, k: int = 5, 
                 backend: str = DEFAULT_BACKEND, db_path: str = None, metadata_backend: str = DEFAULT_METADATA,
                 num_cache_vectors: int = DEFAULT_NUM_CACHE_VECTORS, hot_vector_ids: list[int] = None,
                 compression_option: str = DEFAULT_COMPRESSION):        
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype 
        self.k = k
        self.backend = backend
        self.metadata_backend = metadata_backend
        self.compression_option = compression_option
        self.num_cache_vectors = num_cache_vectors

        if db_path is None:
            if hidden_size == 1536:
                db_path = DEFAULT_1_5B_MODEL_DB_PATH
            elif hidden_size == 2048.:
                db_path = DEFAULT_3B_MODEL_DB_PATH
            elif hidden_size == 3584:
                db_path = DEFAULT_7B_MODEL_DB_PATH
            else:
                raise ValueError(f"No default DB path for hidden size {hidden_size}. Please provide a db_path.")

        # optional compressed storage
        self.cache_matrices = []
        self.pca_W = []
        self.cache_to_real_id = [] # maps compressed matrix indices back to real FAISS IDs

        # load FAISS DB if any of these backends require it
        if self.backend in ["ivfpq", "ivfpq_centroids", "hot_cache"] or self.metadata_backend == "ivfpq_store":
            self.faiss_db = SkippingVectorDB.load(
                folder_path=db_path,
                n_checkpoints=num_blocks-1, # we have one index per block except the last one
                vector_dim=hidden_size,
                device=str(device) if "cuda" in str(device) else "cpu",
                n_probe=DEFAULT_N_PROBE,
            )
            logger.info_rank0(f"Initialised SkippingDB with IVFPQ backend from {db_path}.")

        # distill to cache
        if self.backend in ["cache", "ivfpq_centroids", "hot_cache"]:
            for block_idx in range(num_blocks - 1): # ignore last block since it can't skip
                # distillation strategies
                if self.backend == "cache":
                    # random vectors
                    cache_tensor = torch.randn(self.num_cache_vectors, hidden_size, device=device, dtype=dtype)
                    cache_tensor = F.normalize(cache_tensor, p=2.0, dim=1)
                elif self.backend == "ivfpq_centroids":
                    # extract the coarse quantiser centroids from the loaded FAISS Index
                    cpu_index = faiss.index_gpu_to_cpu(self.faiss_db.indexes[block_idx]) if "cuda" in str(device) else self.faiss_db.indexes[block_idx]
                    centroids = cpu_index.quantizer.reconstruct_n(0, cpu_index.nlist)
                    cache_tensor = torch.as_tensor(centroids, device=device, dtype=torch.float32)
                    cache_tensor = F.normalize(cache_tensor, p=2.0, dim=1).to(dtype)
                    
                    # map the centroid matrix index to a representative ID using the mode of the cluster's skip counts
                    invlists = cpu_index.invlists
                    id_map = {}
                    block_metadata = self.faiss_db.metadata[block_idx]
                    for i in range(cpu_index.nlist):
                        list_size = invlists.list_size(i)
                        if list_size > 0:
                            # fast extraction of FAISS inverted list IDs into a numpy array
                            cluster_ids = faiss.rev_swig_ptr(invlists.get_ids(i), list_size)
                            # extract all skip counts for this cluster
                            skip_counts = [block_metadata[int(vid)].skip_count for vid in cluster_ids if int(vid) in block_metadata]
                            if not skip_counts:
                                id_map[i] = 0
                                continue
                            # find the most common skip decision (the mode)
                            counts = np.bincount(skip_counts)
                            mode_val = np.argmax(counts)
                            # assign the first vector ID in this cluster that perfectly matches the mode
                            # this ensures the CPU logic natively fetches the correct aggregated skip count
                            match_idx = next(vid for vid in cluster_ids if block_metadata[int(vid)].skip_count == mode_val)
                            id_map[i] = int(match_idx)
                        else:
                            id_map[i] = 0 # fallback if empty cluster
                    self.cache_to_real_id.append(id_map)
                    
                elif self.backend == "hot_cache":
                    # extract the most frequently accessed actual vectors
                    cpu_idx = faiss.index_gpu_to_cpu(self.faiss_db.indexes[block_idx]) if "cuda" in str(device) else self.faiss_db.indexes[block_idx]
                    cpu_idx.make_direct_map() # required to reconstruct exact vectors from IVFPQ
                    
                    if not hot_vector_ids:
                        # just take first M vectors if no hot vector list is provided (simulating a hot vector list for benchmarking)
                        hot_vector_ids = list(self.faiss_db.metadata[block_idx].keys())
                    valid_ids = hot_vector_ids[:self.num_cache_vectors]
                    hot_vectors = [torch.as_tensor(cpu_idx.reconstruct(int(vid))) for vid in valid_ids]
                    
                    cache_tensor = torch.stack(hot_vectors).to(device=device, dtype=torch.float32)
                    cache_tensor = F.normalize(cache_tensor, p=2.0, dim=1).to(dtype)
                    
                    # map the local cache index (0...M) back to the real FAISS vector ID
                    id_map = {i: vid for i, vid in enumerate(valid_ids)}
                    self.cache_to_real_id.append(id_map)
                
                # compression strategies
                if self.compression_option == "pca":
                    # fit PCA projection to 128 dims to reduce bandwidth
                    _, _, V = torch.pca_lowrank(cache_tensor.float(), q=128)
                    pca_proj = V.to(dtype=dtype)
                    self.pca_W.append(pca_proj)
                    reduced = torch.matmul(cache_tensor, pca_proj)
                    self.cache_matrices.append(reduced)
                elif self.compression_option == "int8":
                    # proxies W8A16 native compression bandwidth savings
                    # scale by 127 to fill int8 range [-127, 127], round to nearest integer, then cast
                    scaled_tensor = (cache_tensor * 127.0).round().to(torch.int8)
                    self.cache_matrices.append(scaled_tensor)
                else: # normal
                    self.cache_matrices.append(cache_tensor)
                    
            logger.info_rank0(f"Initialised {self.backend} with '{self.compression_option}' optimisation.")

        # CPU component: metadata store per block
        # maps centroid ID -> number of blocks to skip
        self.cpu_metadata = []
        for _ in range(num_blocks - 1):
            # random metadata for valid skips - early exit and mid-sized skips are most likely
            # we have random metadata for benchmarking purposes
            metadata = {
                i: random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.15, 0.3, 0.15, 0.3])[0] 
                for i in range(self.num_cache_vectors)
            }
            self.cpu_metadata.append(metadata)


    def search_gpu(self, block_idx: int, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs strictly on the GPU. Returns (scores, ids)."""
        batch_size = hidden_states.shape[0]
        
        # the final block cannot skip anywhere, return 0s if queried
        if block_idx >= self.num_blocks - 1:
            logger.warning_rank0(f"Block {block_idx} queried for skipping decisions, but it's the final block. Returning 0 scores and -1 IDs.")
            return torch.zeros((batch_size, self.k), dtype=torch.float32, device=self.device), \
                   torch.zeros((batch_size, self.k), dtype=torch.int64, device=self.device)

        if self.backend == "ivfpq":
            # cast to fp32 and ensure contiguous (required for FAISS)
            queries = hidden_states.to(torch.float32).contiguous()
            # l2 normalise on GPU
            queries = F.normalize(queries, p=2.0, dim=-1)
            
            # zero-copy gpu search
            index = self.faiss_db.indexes[block_idx]
            actual_k = min(self.k, index.ntotal)
            
            # FAISS returns PyTorch tensors (scores, ids) directly in VRAM
            if actual_k > 0:
                scores, ids = index.search(queries, actual_k)
            else:
                scores = torch.zeros((batch_size, self.k), dtype=torch.float32, device=self.device)
                ids = torch.zeros((batch_size, self.k), dtype=torch.int64, device=self.device)
            return scores, ids
        
        else: # cache, ivfpq_centroids, hot_cache
            # normalise queries
            queries = torch.nn.functional.normalize(hidden_states, p=2, dim=1)
            
            if self.compression_option == "pca":
                similarities = _fused_pca_search(queries, self.pca_W[block_idx], self.cache_matrices[block_idx])
            elif self.compression_option == "int8":
                similarities = _fused_compressed_search(queries, self.cache_matrices[block_idx])
            else:
                # dense inner product search
                similarities = torch.matmul(queries, self.cache_matrices[block_idx].T) # shape: (batch_size, num_cache_vectors)
            
            # get top K
            scores, ids = torch.topk(similarities, self.k, dim=1)
            return scores, ids


    def get_decision_cpu(self, block_idx: int, ids_cpu: torch.Tensor, scores_cpu: torch.Tensor) -> list[int]:
        """Runs strictly on the CPU. Maps returned IDs to a final routing decision."""
        batch_size = ids_cpu.shape[0]
        
        # the final block cannot skip anywhere, return 0s if queried
        if block_idx >= self.num_blocks - 1:
            logger.warning_rank0(f"Block {block_idx} queried for skipping decisions, but it's the final block. Returning 0 skips.")
            return [0] * batch_size
        
        # select the correct metadata dictionary based on backend
        if self.metadata_backend == "ivfpq_store":
            metadata = self.faiss_db.metadata[block_idx]
            if self.backend == "cache": # cache should use other metadata, but we still safely fallback
                fallback_keys = list(metadata.keys())
        else:
            metadata = self.cpu_metadata[block_idx]
            
        decisions = []
        
        for i in range(batch_size):
            # simple thresholding logic: 
            # currently, we ignore top-1 metadata and just use the hit rate probability to evaluate benchmarks
            top_id = int(ids_cpu[i][0].item())
            top_score = scores_cpu[i][0].item()
            
            # TODO: in tensor-parallel environment we shouldn't use random scores - different ranks can have different skipping decisions. Regardless, we should use a scoring-based system here finally
            if top_id != -1 and top_score*0.0000000001 + random.random() < HIT_RATE_PROBABILITY:
                if self.metadata_backend == "ivfpq_store":
                    if self.backend in ["hot_cache", "ivfpq_centroids"]:
                        # matrix caching returns an index [0...M]. Map it back to the real FAISS ID.
                        real_id = self.cache_to_real_id[block_idx].get(top_id, 0)
                        proposed_skips = metadata[real_id].skip_count
                    elif self.backend == "ivfpq":
                        proposed_skips = metadata[top_id].skip_count
                    else: # cache backend falling back to ivfpq metadata safely
                        real_id = fallback_keys[top_id % len(fallback_keys)]
                        proposed_skips = metadata[real_id].skip_count
                else:
                    # ensure ID is within bounds of simulated metadata
                    top_id = top_id % self.num_cache_vectors 
                    proposed_skips = metadata[top_id]
                    
                max_allowed_skips = max(0, self.num_blocks - block_idx - 1)
                decisions.append(min(proposed_skips, max_allowed_skips))
            else:
                decisions.append(0) # don't skip if confidence is low or ID is invalid
                
        return decisions
