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

from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol


class HasBatchSize(Params):
    batch_size: Param[int] = Param(
        Params._dummy(),
        "batch_size",
        "size of batch for inference",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self) -> None:
        super(HasBatchSize, self).__init__()
        self._setDefault(batch_size=1)

    def getBatchSize(self) -> int:
        return self.getOrDefault(self.batch_size)


class HasBeamWidth(Params):
    beam_width: Param[int] = Param(
        Params._dummy(),
        "beam_width",
        "LLM beam width",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self) -> None:
        super(HasBeamWidth, self).__init__()
        self._setDefault(beam_width=1)

    def getBeamWidth(self) -> int:
        return self.getOrDefault(self.beam_width)


class HasConcurrency(Params):
    concurrency: Param[int] = Param(
        Params._dummy(),
        "concurrency",
        "Number of parallel requests to Triton Inference Server",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self) -> None:
        super(HasConcurrency, self).__init__()
        self._setDefault(concurrency=1)

    def getConcurrency(self) -> int:
        return self.getOrDefault(self.concurrency)


class HasDockerImage(Params):
    docker_image: Param[str] = Param(
        Params._dummy(),
        "docker_image",
        "Docker image for Triton Inference Server",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super(HasDockerImage, self).__init__()

    def getDockerImage(self) -> str:
        return self.getOrDefault(self.docker_image)


class HasModelName(Params):
    model_name: Param[str] = Param(
        Params._dummy(),
        "model_name",
        "Name of model to use in Triton Inference Server",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super(HasModelName, self).__init__()
        self._setDefault(model_name="ensemble")

    def getModelName(self) -> str:
        return self.getOrDefault(self.model_name)


class HasModelPath(Params):
    model_path: Param[str] = Param(
        Params._dummy(),
        "model_path",
        "Host path to model directory for Triton Inference Server",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super(HasModelPath, self).__init__()

    def getModelPath(self) -> str:
        return self.getOrDefault(self.model_path)


class HasOutputLen(Params):
    output_len: Param[int] = Param(
        Params._dummy(),
        "output_len",
        "Output length",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self) -> None:
        super(HasOutputLen, self).__init__()
        self._setDefault(output_len=10)

    def getOutputLen(self) -> int:
        return self.getOrDefault(self.output_len)


class HasPrefix(Params):
    prefix: Param[str] = Param(
        Params._dummy(),
        "prefix",
        "Prompt prefix for LLM",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super(HasPrefix, self).__init__()

    def getPrefix(self) -> str:
        return self.getOrDefault(self.prefix)


class HasProtocol(Params):
    protocol: Param[str] = Param(
        Params._dummy(),
        "protocol",
        "Protocol (http/grpc) used to communicate with Triton Inference Server",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super(HasProtocol, self).__init__()
        self._setDefault(protocol="http")

    def getProtocol(self) -> str:
        return self.getOrDefault(self.protocol)


class HasServer(Params):
    server: Param[str] = Param(
        Params._dummy(),
        "server",
        "Server hostname for Triton Inference Server",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super(HasServer, self).__init__()
        self._setDefault(server="localhost")

    def getServer(self) -> str:
        return self.getOrDefault(self.server)


class HasTokenizer(Params):
    tokenizer: Param[str] = Param(
        Params._dummy(),
        "tokenizer",
        "Tokenizer to use for LLM",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super(HasTokenizer, self).__init__()
        self._setDefault(tokenizer="auto")

    def getTokenizer(self) -> str:
        return self.getOrDefault(self.tokenizer)


class HasTopK(Params):
    topk: Param[int] = Param(
        Params._dummy(),
        "topk",
        "TopK for sampling",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self) -> None:
        super(HasTopK, self).__init__()
        self._setDefault(topk=1)

    def getTopK(self) -> int:
        return self.getOrDefault(self.topk)


class HasTopP(Params):
    topp: Param[float] = Param(
        Params._dummy(),
        "topp",
        "TopP for sampling",
        typeConverter=TypeConverters.toFloat,
    )

    def __init__(self) -> None:
        super(HasTopP, self).__init__()
        self._setDefault(topp=0.0)

    def getTopP(self) -> float:
        return self.getOrDefault(self.topp)


class HasVerbose(Params):
    verbose: Param[bool] = Param(
        Params._dummy(),
        "verbose",
        "verbose logging",
        typeConverter=TypeConverters.toBoolean,
    )

    def __init__(self) -> None:
        super(HasVerbose, self).__init__()
        self._setDefault(verbose=False)

    def getVerbose(self) -> bool:
        return self.getOrDefault(self.verbose)


class LLMParams(
    HasBatchSize,
    HasBeamWidth,
    HasConcurrency,
    HasDockerImage,
    HasInputCol,
    HasModelName,
    HasModelPath,
    HasOutputCol,
    HasOutputLen,
    HasPrefix,
    HasProtocol,
    HasServer,
    HasTokenizer,
    HasTopK,
    HasTopP,
    HasVerbose,
):
    def __init__(self) -> None:
        super().__init__()
        self._setDefault(batch_size=1)
        self._setDefault(inputCol="input")
        self._setDefault(outputCol="output")

    def setBatchSize(self, value: int) -> "LLMParams":
        return self._set(batch_size=value)

    def setBeamWidth(self, value: int) -> "LLMParams":
        return self._set(beam_width=value)

    def setConcurrency(self, value: int) -> "LLMParams":
        return self._set(concurrency=value)

    def setDockerImage(self, value: str) -> "LLMParams":
        return self._set(docker_image=value)

    def setInputCol(self, value: str) -> "LLMParams":
        return self._set(inputCol=value)

    def setModelName(self, value: str) -> "LLMParams":
        return self._set(model_naem=value)

    def setModelPath(self, value: str) -> "LLMParams":
        return self._set(model_path=value)

    def setOutputCol(self, value: str) -> "LLMParams":
        return self._set(outputCol=value)

    def setOutputLen(self, value: int) -> "LLMParams":
        return self._set(output_len=value)

    def setProtocol(self, value: str) -> "LLMParams":
        return self._set(protocol=value)

    def setPrefix(self, value: str) -> "LLMParams":
        return self._set(prefix=value)

    def setServer(self, value: str) -> "LLMParams":
        return self._set(server=value)

    def setTokenizer(self, value: str) -> "LLMParams":
        return self._set(tokenizer=value)

    def setTopK(self, value: int) -> "LLMParams":
        return self._set(topk=value)

    def setTopP(self, value: int) -> "LLMParams":
        return self._set(topp=value)

    def setVerbose(self, value: bool) -> "LLMParams":
        return self._set(verbose=value)
