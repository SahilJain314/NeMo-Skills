# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
import multiprocessing as mp
from enum import Enum

import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.model_config.common import DynamicBatcher
from pytriton.triton import Triton, TritonConfig

from nemo_skills.code_execution import math_grader

ENDPOINT_BIND_ADDRESS = "0.0.0.0"
MAX_BATCH = 9999999

class WorkerSignal(Enum):
    RUN=0
    QUIT=1

def math_check_mp_worker(check_fn: Callable,
                         input_queue: mp.Queue,
                         output_queue: mp.Queue):
    while True:
        signal, idx, args = input_queue.get()
        if signal == WorkerSignal.RUN:
            check_results = check_fn(*args)
            output_queue.put((idx, check_results))
        else:
            return

def extract_and_check(pred_sentence: str,
                      ground_truth: str,
                      extract_from_boxed:bool=True,
                      extract_regex:str=None,
                      **kwargs):
    try:
        pred_output = math_grader.extract_answer(pred_sentence, extract_from_boxed=extract_from_boxed, extract_regex=extract_regex)
        if isinstance(pred_output, str):
            pred_output = pred_output.replace("'''", r'\'\'\'')
            while pred_output.endswith('\\'):
                pred_output = pred_output[:-1]

        if isinstance(ground_truth, str):
            ground_truth = ground_truth.replace("'''", r'\'\'\'')
            while ground_truth.endswith('\\'):
                ground_truth = ground_truth[:-1]
        
        print(pred_sentence, ground_truth)
        return math_grader.math_equal(pred_output,
                                      ground_truth,
                                      include_percentage=kwargs.get("include_percentage", True),
                                      tolerance=kwargs.get("tolerance", 0.0001),
                                      timeout=kwargs.get("timeout", 10)) * 1.0
    except Exception as e:
        print("extract_and_check had an error. Skipping problem (returning incorrect)")
        return False

def lock_method(lock_name):
    """
    Decorator to use in a class to ensure only one method is executing at a time.

    For instance:

        class MyClass:

            def __init__(self):
                self.my_lock = threading.Lock()

            @lock_method("self.my_lock")
            def method1(self):
                return ...

            @lock_method("self.my_lock")
            def method2(self):
                return ...
    """
    # We enforce the usage of the "self." prefix to make it explicit where the lock comes from.
    prefix = "self."
    assert lock_name.startswith(prefix), f"`lock_name` ({lock_name}) must start with '{prefix}'"
    lock_name = lock_name[len(prefix) :]

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            with getattr(self, lock_name):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator

def to_str_list(np_bytes):
    np_bytes = np.array([i[0] for i in np_bytes])
    return [str(i) for i in np.char.decode(np_bytes.astype("bytes"), "utf-8")]

def get_pred_and_gt(inputs):
    inputs['pred_responses'] = to_str_list(inputs['pred_responses'])
    inputs['ground_truth'] = to_str_list(inputs['ground_truth'])
    print(inputs)
    return inputs['pred_responses'], inputs["ground_truth"]

@dataclass
class SimpleMathGrader:
    """
    Class that serves an endpoint to check if a model output is equal to 
    a provided ground-truth answer
    """
    grading_function: Callable
    port: int
    process_count: int
    max_queue_delay_microseconds: int = 2000
    preferred_batch_size: int = 1000

    def __post_init__(self):
        print("POST INIT")
        self.lock = threading.Lock()
        self.inputs = (
            Tensor(name="pred_responses", shape=(-1,), dtype=bytes, optional=False),
            Tensor(name="ground_truth", shape=(-1,), dtype=bytes, optional=False),
        )
        self.outputs = (Tensor(name="rewards", shape=(1,), dtype=np.float32),)
        self.submit_queue = mp.Queue()
        self.result_queue = mp.Queue()
        print("QUEUES BUILT")
        procs = [mp.Process(target=math_check_mp_worker, 
                                   args=(self.grading_function, self.submit_queue, self.result_queue))
                                   for _ in range(self.process_count)]
        print("PROCS BUILT")
        self.workers = []
        for p in procs:
            print("STARTING PROC")
            p.start()
            self.workers.append(p)
        print("WORKERS STARTED")

    @batch
    @lock_method("self.lock")
    def infer(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        predictions, ground_truth = get_pred_and_gt(inputs)
        for inst_idx, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            self.submit_queue.put((WorkerSignal.RUN, inst_idx, (pred, gt)))
        
        rewards = np.zeros(len(ground_truth))
        
        for _ in range(len(ground_truth)):
            inst_idx, reward = self.result_queue.get()
            rewards[inst_idx] = reward

        output_dict = {
            "rewards": rewards.reshape((rewards.shape[0], 1)),
        }

        return output_dict

    def run_server(self):
        triton_config = TritonConfig(
            allow_http=True,
            allow_grpc=False,
            allow_metrics=False,
            http_address=ENDPOINT_BIND_ADDRESS,
            http_port=self.port,
        )

        dynamic_batcher = DynamicBatcher(
            max_queue_delay_microseconds=self.max_queue_delay_microseconds,
            #preferred_batch_size=self.preferred_batch_size,
        )

        # we cut the batch into pieces so we don't need to have a max batch size
        infer_model_config = ModelConfig(batching=True, max_batch_size=MAX_BATCH, batcher=dynamic_batcher)

        with Triton(config=triton_config) as triton:
            print("TRITON STARTUP")
            triton.bind(
                model_name="math_grader",
                infer_func=self.infer,
                inputs=self.inputs,
                outputs=self.outputs,
                config=infer_model_config,
            )
            triton.serve()

