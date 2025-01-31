import bisect
from collections import deque
import csv
from dataclasses import dataclass
import random
import sys
from typing import List, Tuple
import numpy as np

MEM = 120000

@dataclass
class Req:
    prompt_len: int
    ds_output_len: int
    decoded_len: int = 0
    max_output_len: int = 4096
    _sampled: int = None


def get(x: iter, filter_):
    val = next(x)
    while not filter_(val):
        val = next(x)
    return val



PREFILL_INTERVAL = 10

def _calc_max_token_num_needed(cache_len_list: List[Tuple[int, int]]) -> int:
    cache_len_list.sort(key=lambda x: -x[1])

    left_out_len_array = np.array([e[1] for e in cache_len_list])
    has_run_len_array = np.array([e[0] for e in cache_len_list])
    cum_run_len_array = np.cumsum(has_run_len_array)
    size_array = np.arange(1, len(cache_len_list) + 1, 1)

    need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
    return need_max_token_num


def real_mem_requirements(reqs: List[Req]):
    if not reqs:
        return 0
    li = [(req.prompt_len + req.decoded_len, req.ds_output_len - req.decoded_len) for req in reqs]
    return _calc_max_token_num_needed(li)


def sim(generator_, max_reqs=None):
    # sch = FuturePastReqQueue(MEM)
    pending_reqs = []
    batch: List[Req] = []

    _mem_usages = []
    _mem_real_requirements = []
    _overfollows = 0
    _finished = 0
    _decode_steps = 0

    print_ = lambda *_, **__: None
    # print_ = print

    s = generator_

    for i in range(1000000000000000):
        try:

            # if i and i % 10000 == 0:
            #     print(f"{i = }")
            #     print(f"{len(batch) = }")
            #     print(f"{_mem_usages[-1] = }")
            #     print(f"{_finished = }")
            #     # print(f"{batch = }")
            #     # print(f"{[(req.prompt_len, req.decoded_len, req.ds_output_len, req._sampled) for req in batch] = }")
            #     # print(f"{sch.history_output_len = }")
            #     print()

            if max_reqs and _finished > max_reqs:
                break

            # decode steps
            print_("\ndecode")
            for _ in range(PREFILL_INTERVAL):
                _decode_steps += 1
                for req in batch:
                    req.decoded_len += 1
                    if not req.decoded_len < req.ds_output_len:
                        # finished
                        _finished += 1
                        # sch.history_output_len.append(req.decoded_len)
                        print_("finished", req.decoded_len)

                batch = [req for req in batch if req.decoded_len < req.ds_output_len]

                total_req_mem = sum(req.prompt_len + req.decoded_len for req in batch)
                print_(total_req_mem)
                _mem_usages.append(total_req_mem)
                while total_req_mem > MEM:
                    print_("mem boooooom")
                    _overfollows += 1
                    pending_reqs.insert(0, batch.pop())
                    total_req_mem = sum(req.prompt_len + req.decoded_len for req in batch)

            # prefill step
            print_("\nprefill")
            # print_(f"{sch.history_output_len}")
            print_(f"{len(batch) = }")
            # while True:
            MAX_ONCE_PREFILL_TOKENS = 32768
            # sch._init_cache_list(batch)
            prefilled_tokens = 0
            while prefilled_tokens < MAX_ONCE_PREFILL_TOKENS:
                if not pending_reqs:
                    r = next(s)
                    pending_reqs.append(Req(
                        prompt_len=int(r['Request tokens']),
                        ds_output_len=int(r['Response tokens']),
                    ))
                new_req_prefill_len = pending_reqs[0].prompt_len
                total_req_mem = sum(req.prompt_len + req.decoded_len for req in batch)
                if total_req_mem + new_req_prefill_len > MEM * (1 - 0.01):
                    break
                batch.append(pending_reqs.pop(0))
                prefilled_tokens += batch[-1].prompt_len


            total_req_mem = sum(req.prompt_len + req.decoded_len for req in batch)
            print_(f"after prefill {len(batch) = }")
            print_(total_req_mem)
            _mem_usages.append(total_req_mem)
            _mem_real_requirements.append(real_mem_requirements(batch))

            # if _mem_real_requirements[-1] > MEM:
            #     print(_mem_real_requirements[-1], 'vs', MEM * (1 - sch.REVERSED))

        except StopIteration:
            break
        except Exception:
            print(batch)
            print("ERROR!")
            sys.exit()
            break


    _mem_real_requirements = np.array(_mem_real_requirements)
    print(f"Decoding Steps: {_decode_steps}")
    print(f"Memory Utilization: {np.average(_mem_usages) / MEM * 100:.2f} %")
    print(f"Peak Memory: {np.average(_mem_real_requirements / MEM * 100):.2f} %")
    print(f"Evicted Reqs {_overfollows / _finished * 100:.2f} %")



def dist1():
    for _ in range(3000):
        yield {
            'Request tokens': random.randint(32, 4096),
            'Response tokens': random.randint(2048, 4096),
        }
def dist2():
    for _ in range(3000):
        yield {
            'Request tokens': random.randint(3000, 5000),
            'Response tokens': random.randint(3000, 5000),
        }
def dist3():
    for _ in range(3000):
        yield {
            'Request tokens': random.randint(2048, 4096),
            'Response tokens': random.randint(32, 4096),
        }

# random.seed(0)
print("\n\n === Distribution-1 (Decode-heavy)")
sim(dist1(), max_reqs=None)
print("\n\n === Distribution-2 (Balanced)")
sim(dist2(), max_reqs=None)
print("\n\n === Distribution-3 (Prefill-heavy)")
sim(dist3(), max_reqs=None)
