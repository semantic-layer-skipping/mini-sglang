import torch
import random
import torch.nn.functional as F
import faiss.contrib.torch_utils  # enables zero-copy PyTorch tensors in FAISS

from minisgl.utils import init_logger
# Import the SkippingVectorDB we built previously
from minisgl.engine.skipping_vector_db import SkippingVectorDB

logger = init_logger(__name__)

DEFAULT_K = 5
HIT_RATE_PROBABILITY = 0.3 
# 1.5B model
#DEFAULT_DB_PATH = "/home/yff23/data/semantic-layer-skipping/experiments/batch_20260507_154513_Qwen2.5-1.5B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24/db_ivfpq_subsampled_100pct"
# 3B model
DEFAULT_DB_PATH = "/home/yff23/data/semantic-layer-skipping/experiments/batch_20260516_232926_Qwen2.5-3B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24-28-32/db_ivfpq_subsampled_100pct"
# 7B model
#DEFAULT_DB_PATH = "/home/yff23/data/semantic-layer-skipping/experiments/batch_20260514_024813_Qwen2.5-7B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24/db_ivfpq_subsampled_100pct"


DEFAULT_BACKEND = "cache" # ivfpq, cache
DEFAULT_METADATA = "distribution" # distribution, ivfpq_store
DEFAULT_N_PROBE = 64

class VectorCache:

    def __init__(self, num_blocks: int, hidden_size: int, device: torch.device, dtype: torch.dtype, k: int = 5, backend: str = DEFAULT_BACKEND, db_path: str = None, metadata_backend: str = DEFAULT_METADATA):        
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype 
        self.k = k
        self.backend = backend
        self.metadata_backend = metadata_backend
        
        # simulated size of the cache/index per block
        self.num_centroids = 1000

        if self.backend == "ivfpq":
            if db_path is None:
                db_path = DEFAULT_DB_PATH
            self.faiss_db = SkippingVectorDB.load(
                folder_path=db_path,
                n_checkpoints=num_blocks-1, # we have one index per block except the last one
                vector_dim=hidden_size,
                device=str(device).replace("cuda:", "cuda") if "cuda" in str(device) else "cpu",
                n_probe=DEFAULT_N_PROBE,
            )
            logger.info_rank0(f"Initialised SkippingDB with IVFPQ backend from {db_path}.")
        else:
            # vector cache for each block
            self.gpu_indices = []
            for _ in range(num_blocks):
                # random vectors, normalised for cosine similarity
                centroids = torch.randn(self.num_centroids, hidden_size, device=device, dtype=dtype)
                centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
                self.gpu_indices.append(centroids)
            logger.info_rank0(f"Initialised SkippingDB with {num_blocks} blocks, each with {self.num_centroids} centroids of dimension {hidden_size} of type {dtype}.")

        # CPU component: metadata store per block
        # maps centroid ID -> number of blocks to skip
        self.cpu_metadata = []
        for _ in range(num_blocks):
            # random metadata for valid skips - early exit and mid-sized skips are most likely
            # we have random metadata for benchmarking purposes
            metadata = {
                i: random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.15, 0.3, 0.15, 0.3])[0] 
                for i in range(self.num_centroids)
            }
            self.cpu_metadata.append(metadata)

    def search_gpu(self, block_idx: int, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs strictly on the GPU. Returns (scores, ids)."""
        if self.backend == "ivfpq":
            # cast to fp32 and ensure contiguous (required for FAISS)
            queries = hidden_states.to(torch.float32).contiguous()
            # l2 normalise on GPU
            queries = F.normalize(queries, p=2.0, dim=-1)
            
            # zero-copy gpu saerch
            index = self.faiss_db.indexes[block_idx]
            actual_k = min(self.k, index.ntotal)
            
            # FAISS returns PyTorch tensors (scores, ids) directly in VRAM
            if actual_k > 0:
                scores, ids = index.search(queries, actual_k)
            else:
                batch_size = hidden_states.shape[0]
                scores = torch.zeros((batch_size, self.k), dtype=torch.float32, device=self.device)
                ids = torch.zeros((batch_size, self.k), dtype=torch.int64, device=self.device)
            return scores, ids
        
        else:
            # normalise queries
            queries = torch.nn.functional.normalize(hidden_states, p=2, dim=1)
            index = self.gpu_indices[block_idx]
            
            # inner product search
            similarities = torch.matmul(queries, index.T) # shape: (batch_size, num_centroids)
            
            # get top K
            scores, ids = torch.topk(similarities, self.k, dim=1)
            return scores, ids

    def get_decision_cpu(self, block_idx: int, ids_cpu: torch.Tensor, scores_cpu: torch.Tensor) -> list[int]:
        """Runs strictly on the CPU. Maps returned IDs to a final routing decision."""
        batch_size = ids_cpu.shape[0]
        
        # select the correct metadata dictionary based on backend
        if self.metadata_backend == "ivfpq_store":
            metadata = self.faiss_db.metadata[block_idx]
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
                    # retrieve the actual skip decision from the loaded DB
                    proposed_skips = metadata[top_id].skip_count
                else:
                    # ensure ID is within bounds of simulated metadata
                    top_id = top_id % self.num_centroids 
                    proposed_skips = metadata[top_id]
                    
                max_allowed_skips = max(0, self.num_blocks - block_idx - 1)
                decisions.append(min(proposed_skips, max_allowed_skips))
            else:
                decisions.append(0) # don't skip if confidence is low or ID is invalid
                
        return decisions
