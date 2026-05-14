from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Set, Dict, List

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
    
    # buffer to hold split batches for sequential resubmission
    pending_batches: List[Batch] = field(default_factory=list)

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
            
        # ensure we also purge it from any pending split batches
        for batch in self.pending_batches:
            if req in batch.reqs:
                batch.reqs.remove(req)
                break
        # clean up empty batches
        self.pending_batches = [b for b in self.pending_batches if len(b.reqs) > 0]

    def abort_req(self, uid: int) -> Req | None:
        # check standard queues
        for q in self.virtual_queues.values():
            for req in list(q):
                if req.uid == uid:
                    q.remove(req)
                    return req
                    
        # check pending batches
        for batch in self.pending_batches:
            for req in list(batch.reqs):
                if req.uid == uid:
                    batch.reqs.remove(req)
                    self.pending_batches = [b for b in self.pending_batches if len(b.reqs) > 0]
                    return req
        return None

    @property
    def inflight_tokens(self) -> int:
        count = 0
        for q in self.virtual_queues.values():
            tokens_reserved = (self.page_size - 1) * len(q)
            count += sum(req.remain_len for req in q) + tokens_reserved
            
        for batch in self.pending_batches:
            tokens_reserved = (self.page_size - 1) * len(batch.reqs)
            count += sum(req.remain_len for req in batch.reqs) + tokens_reserved
            
        return count

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
            
        # if we have a pending split batch, hand it to the scheduler immediately
        if self.pending_batches:
            return self.pending_batches.pop(0)
        
        # otherwise, perform deepest-ready-first scheduling
        for block_idx in reversed(range(self.num_blocks)):
            reqs_in_queue = self.virtual_queues[block_idx]
            if not reqs_in_queue:
                continue
                
            # get requests waiting for this block, up to the max batch size
            batch_reqs = list(reqs_in_queue)[:self.max_graph_bs]

            compute_reqs = []
            project_reqs = []

            # perform independent routing for every single request
            if block_idx == 0:
                # we never skip the first block
                compute_reqs = batch_reqs
            else:

                # decide independently for each request whether to skip it or not
                for req in batch_reqs:
                    if random.random() < SKIP_PROB:
                        project_reqs.append(req)
                    else:
                        compute_reqs.append(req)
                
            # remove them all from the queue, as they are now mapped to batches
            for req in batch_reqs:
                self.virtual_queues[block_idx].remove(req)
            
            batches = []
            
            # create isolated batches based on routing decisions
            if compute_reqs:
                b_compute = Batch(reqs=compute_reqs, phase="decode")
                b_compute.block_idx = block_idx
                b_compute.is_project = False
                batches.append(b_compute)
                
            if project_reqs:
                b_project = Batch(reqs=project_reqs, phase="decode")
                b_project.block_idx = block_idx
                b_project.is_project = True
                batches.append(b_project)
                
            if batches:
                # pop the first batch to execute now, and save the rest for the next tick
                first_batch = batches.pop(0)
                self.pending_batches.extend(batches)
                return first_batch
            
        return None

    @property
    def runnable(self) -> bool:
        # the manager is runnable if it has queues OR pending split batches
        return len(self.pending_batches) > 0 or any(len(q) > 0 for q in self.virtual_queues.values())
