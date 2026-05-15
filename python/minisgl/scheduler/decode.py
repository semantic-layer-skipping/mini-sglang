from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Set, Dict

from minisgl.core import Batch, Req

SKIP_PROB = 0.3

@dataclass
class DecodeManager:
    page_size: int
    # blocks of model
    num_blocks: int
    # maximum bach sizes supported by our cuda graphs
    max_graph_bs: int
    # virtual pipeline queues
    virtual_queues: Dict[int, Set[Req]] = field(default_factory=dict)

    def __post_init__(self):
        self.virtual_queues = {i: set() for i in range(self.num_blocks)}
    
    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        for req in reqs:
            if req.can_decode:
                # place the request in the queue for its current block
                self.virtual_queues[req.current_block].add(req)

    def remove_req(self, req: Req) -> None:
        for q in self.virtual_queues.values():
            q.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for q in self.virtual_queues.values():
            for req in list(q):
                if req.uid == uid:
                    q.remove(req)
                    return req
        return None

    @property
    def inflight_tokens(self) -> int:
        count = 0
        for q in self.virtual_queues.values():
            tokens_reserved = (self.page_size - 1) * len(q)
            count += sum(req.remain_len for req in q) + tokens_reserved
        return count

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        
        # deepest ready first scheduling
        for block_idx in reversed(range(self.num_blocks)):
            reqs_in_queue = list(self.virtual_queues[block_idx])
            if not reqs_in_queue:
                continue
            
            batch_reqs = reqs_in_queue[:self.max_graph_bs]

            if block_idx == 0:
                # block 0 is always a full compute to extract the initial features
                is_project = False
                selected_reqs = batch_reqs
            else:
                compute_reqs = []
                project_reqs = []

                # routing based on vector search scores for this block (if available)
                for req in batch_reqs:
                    # if we are in the middle of a current skip, continue skipping
                    if req.skip_blocks_remaining > 0:
                        req.skip_blocks_remaining -= 1
                        project_reqs.append(req)
                        
                    # if we are not skipping, evaluate the latest vector search scores
                    elif req.last_search_scores is not None:
                        
                        # TODO: replace this logic with something more principled, use stored metadata from ids and scores to decide how many blocks to skip
                        mean_score = req.last_search_scores.mean().item()
                        if mean_score*0.0000001 + random.random() < SKIP_PROB:
                            num_remaining_blocks = self.num_blocks - 1 - block_idx
                            num_blocks_to_skip = random.randint(1, num_remaining_blocks)
                            # we will skip next block
                            project_reqs.append(req)
                            # decide how many extra blocks to skip after this current one
                            req.skip_blocks_remaining = num_blocks_to_skip - 1
                        else:
                            # otherwise decide to compute next block as normal
                            compute_reqs.append(req)   
                    else:
                        assert False, "This should not happen! All reqs should have scores by now."

                if project_reqs:
                    is_project = True
                    selected_reqs = project_reqs
                else:
                    is_project = False
                    selected_reqs = compute_reqs
                
            # remove items in selected batch from the queues 
            for req in selected_reqs:
                self.virtual_queues[block_idx].remove(req)

            #print(f"Scheduling batch at block {block_idx} with {len(selected_reqs)} reqs. Project={is_project}")
            
            batch = Batch(reqs=selected_reqs, phase="decode")
            batch.block_idx = block_idx
            batch.is_project = is_project
            return batch
            
        return None

    @property
    def runnable(self) -> bool:
        return any(len(q) > 0 for q in self.virtual_queues.values())
