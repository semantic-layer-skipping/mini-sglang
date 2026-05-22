import gc
import shutil
import json
import os
import random
from dataclasses import dataclass
from enum import Enum

import faiss
import numpy as np

from minisgl.utils import init_logger
logger = init_logger(__name__)

# used for testing - points to a pre-created DB with 6 checkpoints and 1536-dim vectors
STORE_DIR = "/home/yff23/data/semantic-layer-skipping/experiments/batch_20260507_154513_Qwen2.5-1.5B-Instruct_wmt19_train_40000s_128t_strict_strict_match_c4-8-12-16-20-24/db_ivfpq_subsampled_100pct"

N_PROBE = 128  # Number of clusters to search in IVFPQ (if used)
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

        if device == "cuda":
            logger.error("FAISS GPU support not yet implemented in this snippet.")
            pass

        # initialise FAISS Indices
        # TODO: experiment with index types
        # TODO: experiment with dimension reduction
        # TODO: experiment with other similarity metrics (currently cosine via IP)
        self.indexes = [faiss.IndexFlatIP(vector_dim) for _ in range(n_checkpoints)]

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
            raise ValueError(
                f"Checkpoint {checkpoint_idx} out of bounds "
                f"(Max {self.n_checkpoints - 1})"
            )

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
        self, checkpoint_idx: int, query_vector: np.ndarray, k: int = 1
    ) -> list[SearchResult]:
        """
        Searches for the k nearest neighbours in the specified checkpoint index.
        Returns a list of search results containing similarity, skip decision,
        and neighbour id, sorted by closest distance.
        """
        index = self.indexes[checkpoint_idx]
        if index.ntotal == 0:
            return []

        # normalise query for cosine similarity
        faiss.normalize_L2(query_vector)

        # cap k to the total number of vectors to prevent faiss errors
        actual_k = min(k, index.ntotal)

        # search for the top k neighbours
        similarities, indices = index.search(query_vector, k=actual_k)

        results = []
        for i in range(actual_k):
            similarity = similarities[0][i]
            neighbour_id = indices[0][i]

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

            # retrieve the decision made for that neighbour
            decision = self.metadata[checkpoint_idx][neighbour_id]

            # retrieve tag metadata for that neighbour
            tag_meta = self.tag_metadata[checkpoint_idx].get(neighbour_id)

            results.append(
                SearchResult(
                    similarity=float(similarity),
                    decision=decision,
                    neighbour_id=int(neighbour_id),  # convert numpy to python int
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
    def load(cls, folder_path: str, n_checkpoints: int, vector_dim: int):
        """Loads indices and metadata from a folder."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"No DB found at {folder_path}")

        db = cls(n_checkpoints, vector_dim)

        for i in range(n_checkpoints):
            index_path = os.path.join(folder_path, f"ckpt_{i}.index")
            meta_path = os.path.join(folder_path, f"ckpt_{i}_metadata.json")

            if not os.path.exists(index_path) or not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"Missing files for checkpoint {i} in {folder_path}"
                )

            db.indexes[i] = faiss.read_index(index_path)

            if hasattr(db.indexes[i], "nprobe"):
                logger.info(f"Setting nprobe={N_PROBE} for index {i}")
                db.indexes[i].nprobe = N_PROBE

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

        logger.info(f"SkippingVectorDB loaded from {folder_path}")
        return db

    @staticmethod
    def create_merged_subsampled_db_from_chunks(
        base_dir: str,
        output_dir: str,
        n_checkpoints: int,
        vector_dim: int,
        keep_fraction: float = 0.10,
    ):
        logger.info(
            f"Creating merged subsampled DB (keeping {keep_fraction * 100}% of vectors)"
        )

        # initialise new DB
        merged_db = SkippingVectorDB(n_checkpoints, vector_dim)

        chunk_dirs = sorted(
            [
                os.path.join(base_dir, d)
                for d in os.listdir(base_dir)
                if d.startswith("db_chunk_")
            ]
        )
        assert chunk_dirs, f"No chunk directories found in {base_dir}"

        for chunk_dir in chunk_dirs:
            logger.info(f"Processing {chunk_dir}...")
            chunk_db = SkippingVectorDB.load(chunk_dir, n_checkpoints, vector_dim)

            for ckpt in range(n_checkpoints):
                index = chunk_db.indexes[ckpt]
                checkpoint_metadata = chunk_db.metadata[ckpt]

                n_vectors = index.ntotal
                if n_vectors == 0:
                    continue

                # select a subset of indices to keep
                n_keep = int(n_vectors * keep_fraction)
                keep_indices = random.sample(range(n_vectors), n_keep)

                # reconstruct vectors (FAISS allows this for Flat indices)
                for idx in keep_indices:
                    vec = index.reconstruct(idx).reshape(1, -1)
                    decision = checkpoint_metadata[idx]

                    # retain optional tag metadata
                    tag_meta = chunk_db.tag_metadata[ckpt].get(idx)

                    # add to the new DB
                    merged_db.add_vector(ckpt, vec, decision, tag_metadata=tag_meta)

            # force garbage collection of the large chunk
            del chunk_db

        # save the new compact DB
        merged_db.save(output_dir)
        logger.info(f"Successfully saved merged subsampled DB to {output_dir}")

    @staticmethod
    def create_ivfpq_db_from_exact(
        source_dir: str,
        output_dir: str,
        n_checkpoints: int,
        vector_dim: int,
        nlist: int = 4096,  # number of clusters (Voronoi cells)
        m: int = 64,  # subquantisers (vector_dim 1536 must be divisible by m)
        nbits: int = 8,  # bits per subquantiser (compresses to 1 byte)
    ):
        """
        Converts an exact DB into an IVFPQ DB.
        This is done checkpoint-by-checkpoint to save memory.
        """
        if vector_dim % m != 0:
            raise ValueError(
                f"vector_dim ({vector_dim}) must be divisible by m ({m}) for IVFPQ."
            )

        logger.info(
            f"Starting memory-efficient IVFPQ Conversion. Reading from {source_dir}"
        )
        # ensure output directory does not exist
        assert not os.path.exists(output_dir), (
            f"Folder {output_dir} already exists. Choose a different path or remove it."
        )
        os.makedirs(output_dir)

        for ckpt in range(n_checkpoints):
            exact_index_path = os.path.join(source_dir, f"ckpt_{ckpt}.index")
            exact_meta_path = os.path.join(source_dir, f"ckpt_{ckpt}_metadata.json")
            exact_tag_meta_path = os.path.join(
                source_dir, f"ckpt_{ckpt}_tag_metadata.json"
            )

            if not os.path.exists(exact_index_path) or not os.path.exists(
                exact_meta_path
            ):
                logger.warning(f"Missing files for checkpoint {ckpt}. Skipping.")
                continue

            logger.info(f"Processing Checkpoint {ckpt}")

            # load only this specific checkpoint's index into memory
            exact_index = faiss.read_index(exact_index_path)
            n_vectors = exact_index.ntotal

            if n_vectors == 0:
                logger.info(f"Checkpoint {ckpt} is empty. Saving empty files.")
                faiss.write_index(
                    faiss.IndexFlatIP(vector_dim),
                    os.path.join(output_dir, f"ckpt_{ckpt}.index"),
                )
                shutil.copy2(
                    exact_meta_path,
                    os.path.join(output_dir, f"ckpt_{ckpt}_metadata.json"),
                )
                if os.path.exists(exact_tag_meta_path):
                    shutil.copy2(
                        exact_tag_meta_path,
                        os.path.join(output_dir, f"ckpt_{ckpt}_tag_metadata.json"),
                    )
                del exact_index
                continue

            logger.info(f"Checkpoint {ckpt}: Extracting {n_vectors} vectors...")
            all_vectors = exact_index.reconstruct_n(0, n_vectors)

            # free memory: we have the raw vectors,
            # so we can delete the exact index immediately
            del exact_index
            gc.collect()

            # FAISS requires at least 39 points per centroid.
            MIN_POINTS_PER_CENTROID = 39
            current_nlist = nlist
            if n_vectors < current_nlist * MIN_POINTS_PER_CENTROID:
                current_nlist = max(1, n_vectors // MIN_POINTS_PER_CENTROID)
                logger.warning(
                    f"Checkpoint {ckpt}: Reduced nlist to {current_nlist} "
                    f"to satisfy FAISS constraints."
                )

            logger.info(
                f"Checkpoint {ckpt}: "
                f"Training IVFPQ with nlist={current_nlist}, m={m}..."
            )
            quantizer = faiss.IndexFlatIP(vector_dim)
            ivfpq_index = faiss.IndexIVFPQ(
                quantizer, vector_dim, current_nlist, m, nbits
            )
            ivfpq_index.metric_type = faiss.METRIC_INNER_PRODUCT

            # train on a maximum number of vectors to save time without losing accuracy
            train_subset = all_vectors
            if n_vectors > MAX_NUM_TRAINING_VECTORS:
                idx = np.random.choice(
                    n_vectors, MAX_NUM_TRAINING_VECTORS, replace=False
                )
                train_subset = all_vectors[idx]

            ivfpq_index.train(train_subset)
            del train_subset

            logger.info(f"Checkpoint {ckpt}: Adding encoded vectors to IVFPQ index...")
            ivfpq_index.add(all_vectors)

            # save the newly created IVFPQ index directly to disk
            out_index_path = os.path.join(output_dir, f"ckpt_{ckpt}.index")
            faiss.write_index(ivfpq_index, out_index_path)

            # save metadata: since IDs are preserved 1:1, we just copy the file
            out_meta_path = os.path.join(output_dir, f"ckpt_{ckpt}_metadata.json")
            shutil.copy2(exact_meta_path, out_meta_path)

            # conditionally save tag metadata
            if os.path.exists(exact_tag_meta_path):
                out_tag_meta_path = os.path.join(
                    output_dir, f"ckpt_{ckpt}_tag_metadata.json"
                )
                shutil.copy2(exact_tag_meta_path, out_tag_meta_path)

            logger.info(f"Checkpoint {ckpt}: Saved IVFPQ index and copied metadata.")

            # garbage collection
            del all_vectors
            del ivfpq_index
            del quantizer
            gc.collect()

        logger.info(
            f"Successfully saved memory-efficient IVFPQ conversion to {output_dir}"
        )


def verify_and_set_faiss_threads():
    # check how many allocated CPUs
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    logger.info(f"Hardware Check: There are {cpus} available CPU cores.")

    # faiss threads
    current_faiss_threads = faiss.omp_get_max_threads()
    logger.info(
        f"Hardware Check: "
        f"FAISS OpenMP is defaulting to {current_faiss_threads} threads."
    )

    # set FAISS to use all available cores
    if current_faiss_threads < cpus // 2:
        num_threads = max(1, cpus // 2)
        faiss.omp_set_num_threads(num_threads)
        logger.info(
            f"Hardware Check: "
            f"Set FAISS OpenMP to use {faiss.omp_get_max_threads()} threads."
        )


# example usage:
if __name__ == "__main__":

    loaded_db = SkippingVectorDB.load(
        STORE_DIR, n_checkpoints=6, vector_dim=1536
    )

    vec = np.random.rand(1, 1536).astype("float32")
    loaded_result = loaded_db.search(checkpoint_idx=0, query_vector=vec)
    logger.info(f"Found decision: {loaded_result}")
