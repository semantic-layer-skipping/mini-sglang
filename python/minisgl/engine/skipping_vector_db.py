import gc
import shutil
import json
import os
import random
from dataclasses import dataclass
from enum import Enum

import faiss
import faiss.contrib.torch_utils  # enables zero-copy PyTorch integration, by monkeypatching FAISS index classes to accept torch tensors directly
import numpy as np
import torch
import torch.nn.functional as F

from minisgl.utils import init_logger
logger = init_logger(__name__)

# used for testing - points to a pre-created DB with 6 checkpoints and 1536-dim vectors
STORE_DIR = "/home/yff23/data/semantic-layer-skipping/experiments/batch_20260507_154513_Qwen2.5-1.5B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24/db_ivfpq_subsampled_100pct"

N_PROBE = 64  # Number of clusters to search in IVFPQ (if used)
MAX_NUM_TRAINING_VECTORS = 160_000

class Action(str, Enum):
    CONTINUE = "continue"
    EXIT = "exit"
    SKIP = "skip"

@dataclass
class SkipDecision:
    action: Action
    skip_count: int = 0  # only used if action is SKIP

    def __str__(self):
        if self.action == Action.SKIP:
            return f"SKIP-{self.skip_count}"
        elif self.action == Action.EXIT:
            return "EXIT"
        else:
            return "CONTINUE"

@dataclass
class SearchResult:
    similarity: float
    decision: SkipDecision
    neighbour_id: int
    tag_metadata: dict | None = None

    def __str__(self):
        return (
            (
                f"SearchResult(similarity={self.similarity:.2f}, "
                f"decision={self.decision}, neighbour_id={self.neighbour_id}"
            )
            + (
                f", tag_metadata={self.tag_metadata}"
                if self.tag_metadata is not None
                else ""
            )
            + ")"
        )

class SkippingVectorDB:
    def __init__(self, n_checkpoints: int, vector_dim: int, device: str = "cpu"):
        self.n_checkpoints = n_checkpoints
        self.vector_dim = vector_dim
        self.device = device
        
        self.gpu_res = None
        self.gpu_id = 0

        # initialise GPU resources if requested
        if self.device.startswith("cuda"):
            try:
                self.gpu_res = faiss.StandardGpuResources()
                # parse specific GPU id if provided (e.g., "cuda:1")
                if ":" in self.device:
                    self.gpu_id = int(self.device.split(":")[1])
                logger.info(f"Initialising FAISS on GPU {self.gpu_id}")
            except AttributeError:
                logger.error("faiss-gpu not installed. Falling back to CPU.")
                self.device = "cpu"

        # initialise FAISS Indices
        self.indexes = []
        for _ in range(n_checkpoints):
            idx = faiss.IndexFlatIP(vector_dim)
            if self.device.startswith("cuda") and self.gpu_res is not None:
                idx = faiss.index_cpu_to_gpu(self.gpu_res, self.gpu_id, idx)
            self.indexes.append(idx)

        # metadata storage
        # maps (checkpoint, vector_id) -> SkipDecision
        self.metadata: list[dict[int, SkipDecision]] = [
            {} for _ in range(n_checkpoints)
        ]

        # optional: store additional info about vectors (e.g. sample id, token id)
        # maps (checkpoint, vector_id) -> dict
        self.tag_metadata: list[dict[int, dict]] = [{} for _ in range(n_checkpoints)]

    def get_index_sizes(self) -> dict[int, int]:
        """
        Returns a dictionary mapping checkpoint indices to the total number of
        vectors stored in their respective FAISS indexes.
        """
        return {ckpt_idx: index.ntotal for ckpt_idx, index in enumerate(self.indexes)}

    def add_vector(
        self,
        checkpoint_idx: int,
        vector: np.ndarray,
        decision: SkipDecision,
        tag_metadata: dict | None = None,
    ):
        """
        Adds a vector and its associated skip decision to the DB.
        Optionally stores a dictionary of tag metadata.
        """
        if checkpoint_idx >= self.n_checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_idx} out of bounds")

        # for now, we support fp32 only
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)

        # normalise vector for cosine similarity
        faiss.normalize_L2(vector)

        index = self.indexes[checkpoint_idx]
        current_id = index.ntotal

        # add to index
        index.add(vector)
        self.metadata[checkpoint_idx][current_id] = decision

        # safely store optional tags
        if tag_metadata is not None:
            self.tag_metadata[checkpoint_idx][current_id] = tag_metadata.copy()

    def search(
        self, checkpoint_idx: int, query_vector, k: int = 1
    ) -> list[SearchResult]:
        """
        Searches for the k nearest neighbours in the specified checkpoint index.
        Returns a list of search results containing similarity, skip decision,
        and neighbour id, sorted by closest distance.
        """
        index = self.indexes[checkpoint_idx]
        if index.ntotal == 0:
            return []

        # cap k to the total number of vectors to prevent faiss errors
        actual_k = min(k, index.ntotal)

        # GPU zero-copy path
        if isinstance(query_vector, torch.Tensor):
            # faiss api requires fp32 only, so we cast (on VRAM)
            if query_vector.dtype != torch.float32:
                query_vector = query_vector.to(torch.float32)
                
            # FAISS requires contiguous memory blocks
            if not query_vector.is_contiguous():
                query_vector = query_vector.contiguous()

            # normalise vector for cosine similarity natively on GPU
            query_vector = F.normalize(query_vector, p=2.0, dim=-1)
            
            # search natively using torch tensors
            similarities, indices = index.search(query_vector, actual_k)
            is_tensor = True

        # CPU path with numpy
        else:
            # faiss api requires fp32 only, so we cast
            if query_vector.dtype != np.float32:
                query_vector = query_vector.astype(np.float32)

            faiss.normalize_L2(query_vector)

            # search for the top k neighbours
            similarities, indices = index.search(query_vector, k=actual_k)
            is_tensor = False


        results = []
        for i in range(actual_k):
            # safely extract values whether they are tensors or numpy types
            similarity = similarities[0][i].item() if is_tensor else similarities[0][i]
            neighbour_id = indices[0][i].item() if is_tensor else indices[0][i]

            # faiss returns -1 if it cannot find valid neighbours
            # (e.g. empty clusters in ivf indexes or hitting the nprobe limit)
            if neighbour_id == -1:
                logger.debug(
                    f"FAISS returned -1 for ckpt {checkpoint_idx} at rank {i}. "
                    f"similarity={similarity}. neighbour_id={neighbour_id}. "
                    f"Skipping this and remaining neighbours."
                )
                # if we hit a -1 padding, the remaining results will also be invalid
                break

            # retrieve the decision made for that neighbour (requires CPU integer)
            decision = self.metadata[checkpoint_idx][neighbour_id]

            # retrieve tag metadata for that neighbour
            tag_meta = self.tag_metadata[checkpoint_idx].get(neighbour_id)

            results.append(
                SearchResult(
                    similarity=float(similarity),
                    decision=decision,
                    neighbour_id=int(neighbour_id),
                    tag_metadata=tag_meta,
                )
            )

        return results

    def save(self, folder_path: str):
        """Saves raw indices and metadata to a specific folder using JSON."""
        assert not os.path.exists(folder_path), (
            f"Folder {folder_path} already exists. "
            "Choose a different path or remove it."
        )
        os.makedirs(folder_path)

        for i, (index, meta, tag_meta) in enumerate(
            zip(self.indexes, self.metadata, self.tag_metadata, strict=True)
        ):
            # save index
            index_path = os.path.join(folder_path, f"ckpt_{i}.index")
            
            # FAISS cannot serialise GPU indices, so we pull down to CPU before saving.
            if self.device.startswith("cuda"):
                cpu_index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(cpu_index, index_path)
            else:
                faiss.write_index(index, index_path)

            # save metadata
            # convert {int: SkipDecision} -> {str: dict} for JSON
            json_meta = {str(k): v.__dict__ for k, v in meta.items()}

            meta_path = os.path.join(folder_path, f"ckpt_{i}_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(json_meta, f, indent=2)

            # save tag metadata (only if it has contents)
            if tag_meta:
                json_tag_meta = {str(k): v for k, v in tag_meta.items()}
                tag_meta_path = os.path.join(folder_path, f"ckpt_{i}_tag_metadata.json")
                with open(tag_meta_path, "w") as f:
                    json.dump(json_tag_meta, f, indent=2)

            logger.info(f"Saved index {i} with {index.ntotal} vectors.")

        logger.info(f"SkippingVectorDB content saved to {folder_path}")

    @classmethod
    def load(cls, folder_path: str, n_checkpoints: int, vector_dim: int, device: str = "cpu", n_probe: int = N_PROBE):
        """Loads indices and metadata from a folder."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"No DB found at {folder_path}")

        db = cls(n_checkpoints, vector_dim, device=device)

        for i in range(n_checkpoints):
            index_path = os.path.join(folder_path, f"ckpt_{i}.index")
            meta_path = os.path.join(folder_path, f"ckpt_{i}_metadata.json")

            if not os.path.exists(index_path) or not os.path.exists(meta_path):
                raise FileNotFoundError(f"Missing files for checkpoint {i}")

            # read into cpu index first
            cpu_index = faiss.read_index(index_path)

            if hasattr(cpu_index, "nprobe"):
                cpu_index.nprobe = n_probe

            # if on gpu, push to vram with optimisations
            if db.device.startswith("cuda") and db.gpu_res is not None:
                cloner_options = faiss.GpuClonerOptions()
                cloner_options.useFloat16 = True
                
                # if the index is IVFPQ, and not using inner product, enable precomputed tables for faster searching
                if isinstance(cpu_index, faiss.IndexIVF) and cpu_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                    cloner_options.usePrecomputed = True

                db.indexes[i] = faiss.index_cpu_to_gpu(
                    db.gpu_res, db.gpu_id, cpu_index, cloner_options
                )
            else:
                db.indexes[i] = cpu_index

            with open(meta_path) as f:
                raw_data = json.load(f)

            # reconstruct: {str: dict} -> {int: SkipDecision}
            db.metadata[i] = {}
            for k_str, v_dict in raw_data.items():
                v_dict["action"] = Action(str(v_dict["action"]).lower())
                db.metadata[i][int(k_str)] = SkipDecision(**v_dict)

            # optionally load tag metadata if the file exists (backwards compatibility)
            tag_meta_path = os.path.join(folder_path, f"ckpt_{i}_tag_metadata.json")
            if os.path.exists(tag_meta_path):
                with open(tag_meta_path) as f:
                    raw_tag_data = json.load(f)
                db.tag_metadata[i] = {int(k): v for k, v in raw_tag_data.items()}

        logger.info(f"SkippingVectorDB loaded from {folder_path} on {device.upper()}")
        return db


# example usage:
if __name__ == "__main__":
    # load on cuda
    loaded_db = SkippingVectorDB.load(
        STORE_DIR, n_checkpoints=6, vector_dim=1536, device="cuda"
    )

    logger.info("Testing CPU NumPy Search")
    vec_cpu = np.random.rand(1, 1536).astype("float32")
    loaded_result_cpu = loaded_db.search(checkpoint_idx=0, query_vector=vec_cpu)
    logger.info(f"Found decision (CPU): {loaded_result_cpu}")

    logger.info("Testing GPU PyTorch Search")
    # simulate a bfloat16 hidden state existing directly on the GPU
    vec_gpu = torch.rand(1, 1536, dtype=torch.bfloat16, device="cuda")
    loaded_result_gpu = loaded_db.search(checkpoint_idx=0, query_vector=vec_gpu)
    logger.info(f"Found decision (GPU): {loaded_result_gpu}")
