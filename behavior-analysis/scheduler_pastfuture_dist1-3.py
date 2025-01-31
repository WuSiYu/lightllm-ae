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


class FuturePastReqQueue():
    WINDOW_SIZE = 1000
    MINIMUM_SAMPLES = 200
    MAXIMUM_LISTS = 5
    REVERSED = 0.03

    def __init__(self, max_total_tokens) -> None:
        self.max_total_tokens = max_total_tokens
        initial_len = 512
        self.history_output_len = deque([initial_len] * (self.WINDOW_SIZE // 2), maxlen=self.WINDOW_SIZE)

    def _sample_cache_list(self, reqs: List[Req], samples=1) -> List[List[Tuple[int, int]]]:
        cache_len_lists = [[] for _ in range(samples)]
        his_Lo = sorted(self.history_output_len)
        for req in reqs:
            dl = req.decoded_len
            pos = bisect.bisect(his_Lo, dl)
            sample_range = [dl] + his_Lo[pos:]
            sample_range.append(max(req.max_output_len, sample_range[-1]))

            for i in range(samples):
                random_p = np.random.random() * (len(sample_range)-1)
                l_pos = int(random_p)
                l_val, r_val = sample_range[l_pos:l_pos+2]

                # 线性差值
                sampled = round(l_val + (r_val - l_val) * (random_p - l_pos))
                cache_len_lists[i].append((req.prompt_len + dl, sampled - dl))

        return cache_len_lists

    def _calc_max_token_num_needed(self, cache_len_list: List[Tuple[int, int]]) -> int:
        cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in cache_len_list])
        has_run_len_array = np.array([e[0] for e in cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        return need_max_token_num


    def _init_cache_list(self, reqs: List[Req]):
        if reqs:
            n_lists = min(self.MAXIMUM_LISTS, int(self.MINIMUM_SAMPLES / len(reqs)) + 1)
            self._cache_len_lists = self._sample_cache_list(reqs, samples=n_lists)
        else:
            self._cache_len_lists = [[]]
        self.cache_len_list = self._cache_len_lists[0]   # keep compatibility

    def real_mem_requirements(self, reqs: List[Req]):
        if not reqs:
            return 0
        li = [(req.prompt_len + req.decoded_len, req.ds_output_len - req.decoded_len) for req in reqs]
        return self._calc_max_token_num_needed(li)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req: Req):
        need_max_token_nums = []
        for li in self._cache_len_lists:
            newreq_output_len_sample = random.choice(self.history_output_len)
            li.append((req.prompt_len + req.decoded_len, newreq_output_len_sample))
            need_max_token_nums.append(self._calc_max_token_num_needed(li))
        need_max_token_num = np.max(need_max_token_nums)
        # print(need_max_token_nums)

        ok_token_num = need_max_token_num < self.max_total_tokens * (1 - self.REVERSED)

        return ok_token_num


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
    sch = FuturePastReqQueue(MEM)
    pending_reqs = []
    batch: List[Req] = []

    _mem_usages = []
    _mem_real_requirements = []
    _overfollows = 0
    _finished = 0
    _decode_steps = 0

    print_ = lambda *_, **__: None
    # print_ = print

    # with open(FILE) as f:
    #     s = csv.DictReader(f)
    #     s = filter(filter_func, s)

    #     SKIP = 0
    #     for _ in range(SKIP):
    #         next(s)
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
                        sch.history_output_len.append(req.decoded_len)
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
            sch._init_cache_list(batch)
            prefilled_tokens = 0
            while prefilled_tokens < MAX_ONCE_PREFILL_TOKENS:
                if not pending_reqs:
                    r = next(s)
                    pending_reqs.append(Req(
                        prompt_len=int(r['Request tokens']),
                        ds_output_len=int(r['Response tokens']),
                    ))
                if sch._can_add_new_req(pending_reqs[0]):
                    batch.append(pending_reqs.pop(0))
                    prefilled_tokens += batch[-1].prompt_len
                else:
                    break
                # new_req_prefill_len = pending_reqs[0].prompt_len
                # total_req_mem = sum(req.prompt_len + req.decoded_len for req in batch)
                # if total_req_mem + new_req_prefill_len > MEM * (1 - 0.01):
                #     break
                # batch.append(pending_reqs.pop(0))
                # prefilled_tokens += batch[-1].prompt_len


            total_req_mem = sum(req.prompt_len + req.decoded_len for req in batch)
            print_(f"after prefill {len(batch) = }")
            print_(total_req_mem)
            _mem_usages.append(total_req_mem)
            _mem_real_requirements.append(real_mem_requirements(batch))

            # if _mem_real_requirements[-1] > MEM:
            #     print(_mem_real_requirements[-1], 'vs', MEM * (1 - sch.REVERSED))

        except StopIteration:
            break
        except Exception as e:
            print(batch)
            print("ERROR!")
            # sys.exit()
            raise e
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
