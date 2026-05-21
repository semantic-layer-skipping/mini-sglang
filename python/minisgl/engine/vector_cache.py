import torch
import random
from minisgl.utils import init_logger

logger = init_logger(__name__)

DEFAULT_K = 5
HIT_RATE_PROBABILITY = 0.3

class VectorCache:

    def __init__(self, num_blocks: int, hidden_size: int, device: torch.device, dtype: torch.dtype, k: int = 5):        
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype 
        self.k = k
        
        # simulated size of the cache/index per block
        self.num_centroids = 1000

        # TODO: can try faiss-gpu ivfpq here too
        # GPU component: vector index for each block
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
        metadata = self.cpu_metadata[block_idx]
        decisions = []
        
        for i in range(batch_size):
            # simple thresholding logic: 
            # currently, we ignore top-1 metadata and just use the hit rate probability to evaluate benchmarks
            top_id = int(ids_cpu[i][0].item())
            top_score = scores_cpu[i][0].item()
            
            # TODO: in tensor-parallel environment we shuoldn't use random scores - different ranks can have different skipping decisions. Regardless, we should use a scoring-based system here finally
            if top_score*0.0000000001 + random.random() < HIT_RATE_PROBABILITY:
                proposed_skips = metadata[top_id]
                max_allowed_skips = max(0, self.num_blocks - block_idx - 1)
                decisions.append(min(proposed_skips, max_allowed_skips))
            else:
                decisions.append(0) # don't skip if confidence is low
                
        return decisions
