#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
#

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.functions import predict_batch_udf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import struct, udf
from pyspark.sql.types import StringType
from tritonclient._client import InferenceServerClientBase
from tritonclient.grpc import InferInput as InferInputGRPC
from tritonclient.grpc import InferResult as InferResultGRPC
from tritonclient.http import InferInput as InferInputHTTP
from tritonclient.http import InferResult as InferResultHTTP
from tritonclient.utils import np_to_triton_dtype

from .llm_params import LLMParams

InferInput = Union[InferInputGRPC, InferInputHTTP]
InferResult = Union[InferResultGRPC, InferResultHTTP]


@dataclass
class Options:
    num_executors: int
    batch_size: int
    beam_width: int
    concurrency: int
    model_name: str
    output_len: int
    protocol: str
    server: str
    topk: int
    topp: float
    verbose: bool


def create_inference_server_client(
    protocol: str, url: str, concurrency: int, verbose: bool
) -> InferenceServerClientBase:
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(
            url, concurrency=concurrency, verbose=verbose
        )
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url, verbose=verbose)


def send_requests(
    model_name: str,
    inputs: List[InferInput],
    client: InferenceServerClientBase,
    request_parallelism: int,
) -> InferResult:
    results = []
    for _ in range(request_parallelism):
        result = client.infer(model_name, inputs)
        results.append(result)
    return results


def prepare_tensor(name: str, input: np.ndarray, protocol: str) -> InferInput:
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def prepare_inputs(input0: np.ndarray, FLAGS: Options) -> List[InferInput]:
    bad_words_list = np.array([[""] for x in input0], dtype=object)
    stop_words_list = np.array([[""] for x in input0], dtype=object)
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.uint32) * FLAGS.output_len
    runtime_top_k = (FLAGS.topk * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
    runtime_top_p = FLAGS.topp * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    temperature = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    repetition_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    random_seed = 0 * np.ones([input0_data.shape[0], 1]).astype(np.uint64)
    output_log_probs = True * np.ones([input0_data.shape[0], 1]).astype(bool)
    beam_width = (FLAGS.beam_width * np.ones([input0_data.shape[0], 1])).astype(
        np.uint32
    )
    pad_ids = 2 * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    end_ids = 2 * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    min_length = 1 * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    presence_penalty = 0.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    inputs = [
        prepare_tensor("INPUT_0", input0_data, FLAGS.protocol),
        prepare_tensor("INPUT_1", output0_len, FLAGS.protocol),
        prepare_tensor("INPUT_2", bad_words_list, FLAGS.protocol),
        prepare_tensor("INPUT_3", stop_words_list, FLAGS.protocol),
        prepare_tensor("pad_id", pad_ids, FLAGS.protocol),
        prepare_tensor("end_id", end_ids, FLAGS.protocol),
        prepare_tensor("beam_width", beam_width, FLAGS.protocol),
        prepare_tensor("runtime_top_k", runtime_top_k, FLAGS.protocol),
        prepare_tensor("runtime_top_p", runtime_top_p, FLAGS.protocol),
        prepare_tensor("temperature", temperature, FLAGS.protocol),
        prepare_tensor("len_penalty", len_penalty, FLAGS.protocol),
        prepare_tensor("repetition_penalty", repetition_penalty, FLAGS.protocol),
        prepare_tensor("min_length", min_length, FLAGS.protocol),
        prepare_tensor("presence_penalty", presence_penalty, FLAGS.protocol),
        prepare_tensor("random_seed", random_seed, FLAGS.protocol),
        prepare_tensor("output_log_probs", output_log_probs, FLAGS.protocol),
    ]

    return inputs


def triton_fn(FLAGS: Options) -> Callable[[np.ndarray], np.ndarray]:
    import numpy as np
    from pyspark import TaskContext

    # get executor number to allocate GPUs and ports in a consistent manner
    task_context = TaskContext.get()
    assert task_context, "Unable to obtain TaskContext"
    executor_id = task_context.partitionId() % FLAGS.num_executors

    if FLAGS.server == "localhost":
        # allocate localhost ports by executor_id
        # TODO: choose random ports
        http_port = 3000 + executor_id * 10
        grpc_port = http_port + 1
        port = http_port if FLAGS.protocol == "http" else grpc_port
        url = f"{FLAGS.server}:{port}"
    else:
        # user-provided server should include port
        url = FLAGS.server

    client = create_inference_server_client(
        protocol=FLAGS.protocol,
        url=url,
        concurrency=FLAGS.concurrency,
        verbose=FLAGS.verbose,
    )

    def predict(input_batch: np.ndarray) -> np.ndarray:
        # invoke ensemble model via Triton
        inputs = prepare_inputs(input_batch, FLAGS)
        batch_size = len(input_batch)

        results = send_requests(FLAGS.model_name, inputs, client, request_parallelism=1)

        ensemble_output0 = results[0].as_numpy("OUTPUT_0")
        ensemble_output0 = ensemble_output0.reshape([-1, batch_size]).T.tolist()
        ensemble_output0 = [
            [char.decode("UTF-8") for char in line] for line in ensemble_output0
        ]

        # TODO: remove input text
        # return single numpy array
        return np.array(ensemble_output0)

    return predict


class TritonLLM(Model, LLMParams):
    # TODO: factor out generic Triton Server Model
    _input_kwargs: Dict[str, Any]

    @keyword_only
    def __init__(
        self,
        *,
        batch_size: Optional[int] = None,
        beam_width: Optional[int] = None,
        concurrency: Optional[int] = None,
        docker_image: Optional[str] = None,
        inputCol: Optional[str] = None,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        outputCol: Optional[str] = None,
        output_len: Optional[int] = None,
        prefix: Optional[str] = None,
        protocol: Optional[str] = None,
        server: Optional[str] = None,
        tokenizer: Optional[str] = None,
        topk: Optional[int] = None,
        topp: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        super().__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

        self._spark = SparkSession.getActiveSession()
        if self._spark:
            self._sc = self._spark.sparkContext
            self._num_executors = self._infer_num_executors()

    def _infer_num_executors(self) -> int:
        assert self._spark
        executor_instances = self._spark.conf.get("spark.executor.instances", "-1")
        assert executor_instances
        num_executors = int(executor_instances)
        if num_executors == -1:
            jsc = self._spark.sparkContext._jsc.sc()
            num_executors = len(jsc.statusTracker().getExecutorInfos()) - 1
        return num_executors

    def _transform(self, dataset: DataFrame) -> DataFrame:
        # convert ML Params to simple dataclass
        FLAGS = Options(
            num_executors=self._num_executors,
            batch_size=self.getBatchSize(),
            beam_width=self.getBeamWidth(),
            concurrency=self.getConcurrency(),
            model_name=self.getModelName(),
            output_len=self.getOutputLen(),
            protocol=self.getProtocol(),
            server=self.getServer(),
            topk=self.getTopK(),
            topp=self.getTopP(),
            verbose=self.getVerbose(),
        )

        # prepend prefix to data, if defined
        if self.isDefined(self.prefix):
            # TODO: add prefix without modifying dataset?
            prefix = self.getPrefix()

            @udf(StringType())
            def instruct(text: str) -> str:
                return f"<s>[INST] <<SYS>>\n{prefix}\n</SYS>>\n\n{text}[/INST]"

            dataset = (
                dataset.withColumn("_tmp", instruct(self.getInputCol()))
                .select("_tmp")
                .withColumnRenamed("_tmp", self.getInputCol())
            )

        # ensure mininum number of partitions
        if dataset.rdd.getNumPartitions() < self._num_executors:
            dataset = dataset.repartition(self._num_executors)

        # create pandas_udf for generation via Triton
        generate = predict_batch_udf(
            partial(triton_fn, FLAGS),
            return_type=StringType(),
            input_tensor_shapes=[[1]],
            batch_size=8,
        )

        # return dataset w/ generated column
        return dataset.withColumn(
            self.getOutputCol(), generate(struct(self.getInputCol()))
        )

    def startServers(self, world_size: int = 1) -> None:
        import time

        import docker
        import tritonclient.grpc as grpcclient

        assert (
            self.getServer() == "localhost"
        ), "Cannot invoke 'startServers' for remote servers."
        print(f"starting {self._num_executors} server(s).")

        assert self.isDefined(
            self.docker_image
        ), "Please use 'setDockerImage' to configure an image for localhost servers."
        docker_image = self.getDockerImage()

        def start_triton(
            it: Iterable[int], world_size: int = world_size
        ) -> Iterable[bool]:
            # get executor number to allocate GPUs and ports in a consistent manner
            # from pyspark import TaskContext
            # task_context = TaskContext.get()
            # executor_id = task_context.partitionId() % num_executors
            executor_id = next(iter(it))
            print(f">>>> executor_id: {executor_id}")

            # allocate ports by executor_id
            # TODO: choose random ports
            http_port = 3000 + executor_id * 10
            grpc_port = http_port + 1
            metrics_port = http_port + 2

            gpus = range(
                executor_id * world_size, executor_id * world_size + world_size
            )
            gpu_str = ",".join([str(x) for x in gpus])

            client = docker.from_env()
            # TODO: user-defined label?
            containers = client.containers.list(filters={"label": "spark-triton"})
            if containers:
                print(">>>> containers: {}".format([c.short_id for c in containers]))
            else:
                # launch container on a GPU id matching the executor_id (for now, since I'm on a DGX host w/ 16 GPUs)
                # mapping ports by executor_id
                # TODO: build runtime container and mount model directory
                container = client.containers.run(
                    docker_image,
                    f"bash -c 'python3 scripts/launch_triton_server.py --world_size={world_size} --model_repo=all_models/gpt; while true; do sleep 10; done'",
                    detach=True,
                    device_requests=[
                        docker.types.DeviceRequest(
                            device_ids=[gpu_str], capabilities=[["gpu"]]
                        )
                    ],
                    labels={"spark-triton": f"{executor_id}"},
                    network_mode="bridge",
                    remove=True,
                    ports={8000: http_port, 8001: grpc_port, 8002: metrics_port},
                    shm_size="8g",
                    volumes={
                        "/home/leey/dev/tekit_backend": {
                            "bind": "/tensorrt_llm_backend",
                            "mode": "rw",
                        },
                        "/raid": {
                            "bind": "/raid",
                            "mode": "rw",
                        },
                    },
                    working_dir="/tensorrt_llm_backend",
                )
                print(">>>> starting triton: {}".format(container.short_id))

                # wait for triton to be ready
                time.sleep(15)
                client = grpcclient.InferenceServerClient(f"localhost:{grpc_port}")
                ready = False
                while not ready:
                    try:
                        ready = client.is_server_ready()
                    except Exception as e:
                        time.sleep(5)

            return [True]

        nodeRDD = self._sc.parallelize(
            list(range(self._num_executors)), self._num_executors
        )
        nodeRDD.barrier().mapPartitions(start_triton).collect()

    def stopServers(self) -> None:
        assert (
            self.getServer() == "localhost"
        ), "Cannot invoke 'stopServers' for remote servers."
        print(f"stopping {self._num_executors} server(s)")

        num_executors = self._num_executors

        def stop_triton(it: Iterable[int]) -> Iterable[bool]:
            import docker

            # get executor number to allocate GPUs and ports in a consistent manner
            from pyspark import TaskContext

            task_context = TaskContext.get()
            assert task_context, "Unable to obtain TaskContext"
            executor_id = task_context.partitionId() % num_executors

            client = docker.from_env()
            containers = client.containers.list(filters={"label": "spark-triton"})
            if containers:
                container = containers[executor_id]
                print(f">>>> stopping triton: {container}")
                container.stop(timeout=120)

            return [True]

        nodeRDD = self._sc.parallelize(range(self._num_executors), self._num_executors)
        nodeRDD.barrier().mapPartitions(stop_triton).collect()
