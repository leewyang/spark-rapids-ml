#
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import base64
import struct
from array import array
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyspark.ml.common import _py2java
from pyspark.ml.feature import MinMaxScalerModel as SparkMinMaxScalerModel
from pyspark.ml.feature import _MinMaxScalerParams
from pyspark.ml.linalg import DenseVector, Vector
from pyspark.ml.param.shared import HasInputCols
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    Row,
    StringType,
    StructField,
    StructType,
)

from ..core import (
    CumlT,
    FitInputType,
    _ConstructFunc,
    _CumlEstimator,
    _CumlModelWithColumns,
    _EvaluateFunc,
    _TransformFunc,
    param_alias,
    transform_evaluate,
)
from ..params import P, _CumlClass, _CumlParams
from ..utils import (
    _concat_and_free,
    _get_spark_session,
    dtype_to_pyspark_type,
    java_uid,
)


class MinMaxScalerClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {}

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {"feature_range": (0.0, 1.0), "copy": True}


class _MinMaxScalerCumlParams(_CumlParams, _MinMaxScalerParams, HasInputCols):
    """
    Shared Spark Params for MinMaxScaler and MinMaxScalerModel.
    """

    def setInputCol(self: P, value: Union[str, List[str]]) -> P:
        """
        Sets the value of :py:attr:`inputCol` or :py:attr:`inputCols`. Used when input vectors are stored in a single column.
        """
        if isinstance(value, str):
            self.set_params(inputCol=value)
        else:
            self.set_params(inputCols=value)
        return self

    def setInputCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`inputCols`. Used when input vectors are stored as multiple feature columns.
        """
        return self.set_params(inputCols=value)

    def setOutputCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`outputCol`
        """
        return self.set_params(outputCol=value)


class MinMaxScaler(MinMaxScalerClass, _CumlEstimator, _MinMaxScalerCumlParams):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def _get_cuml_fit_func(
        self,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[[FitInputType, Dict[str, Any]], Dict[str, Any],]:
        array_order = self._fit_array_order()

        def _cuml_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            import cupy as cp
            from cuml.preprocessing import MinMaxScaler as CumlMinMaxScaler
            from cupy.cuda import nccl
            from pyspark import BarrierTaskContext

            taskcontext = BarrierTaskContext.get()
            rank = params[param_alias.rank]
            ndev = params[param_alias.nranks]

            # get NCCL communicator unique id for cluster
            nccl_uid = base64.b64encode(array("b", nccl.get_unique_id())).decode(
                "utf-8"
            )
            nccl_uids = taskcontext.allGather(nccl_uid)
            commId = struct.unpack("@128b", base64.b64decode(nccl_uids[0]))

            # local fit
            # concated = _concat_and_free(dfs, order=array_order)
            data = [x for x, _, _ in dfs]
            concated = _concat_and_free(data, order=array_order)
            mms = CumlMinMaxScaler()
            mms.fit(concated)

            # allReduce to compute global min/max
            comm = nccl.NcclCommunicator(ndev, commId, rank)

            data_min = cp.array(mms.data_min_)
            data_max = cp.array(mms.data_max_)
            data_size = len(data_min) * data_min.dtype.itemsize

            stream = cp.cuda.stream.get_current_stream()
            comm.allReduce(
                data_min.data.ptr,
                data_min.data.ptr,
                data_size,
                nccl.NCCL_FLOAT32,
                nccl.NCCL_MIN,
                stream.ptr,
            )
            comm.allReduce(
                data_max.data.ptr,
                data_max.data.ptr,
                data_size,
                nccl.NCCL_FLOAT32,
                nccl.NCCL_MAX,
                stream.ptr,
            )
            stream.synchronize()

            return {
                "data_max_": [data_max.tolist()],
                "data_min_": [data_min.tolist()],
                "n_cols": params[param_alias.num_cols],
                "dtype": data_min.dtype.name,
            }

        return _cuml_fit

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("data_max_", ArrayType(FloatType(), False), False),
                StructField("data_min_", ArrayType(DoubleType(), False), False),
                StructField("n_cols", IntegerType(), False),
                StructField("dtype", StringType(), False),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> "MinMaxScalerModel":
        return MinMaxScalerModel.from_row(result)


class MinMaxScalerModel(
    MinMaxScalerClass, _CumlModelWithColumns, _MinMaxScalerCumlParams
):
    def __init__(
        self,
        data_max_: List[float],
        data_min_: List[float],
        n_cols: int,
        dtype: str,
    ):
        super().__init__(
            data_max_=data_max_, data_min_=data_min_, n_cols=n_cols, dtype=dtype
        )

        self.data_max_ = data_max_
        self.data_min_ = data_min_
        self._cpu_model: Optional[SparkMinMaxScalerModel] = None

    @property
    def originalMax(self) -> Vector:
        return DenseVector(self.data_max_)

    @property
    def originalMin(self) -> Vector:
        return DenseVector(self.data_min_)

    def cpu(self) -> SparkMinMaxScalerModel:
        """Return the PySpark ML MinMaxScalerModel"""
        if self._cpu_model is None:
            sc = _get_spark_session().sparkContext
            assert sc._jvm is not None

            java_data_max = _py2java(sc, self.data_max_)
            java_data_min = _py2java(sc, self.data_min_)
            java_model = sc._jvm.org.apache.spark.ml.feature.MinMaxScalerModel(
                java_uid(sc, "mms"), java_data_min, java_data_max
            )
            self._cpu_model = SparkMinMaxScalerModel(java_model)
            self._copyValues(self._cpu_model)

        return self._cpu_model

    def _get_cuml_transform_func(
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc],]:
        cuml_alg_params = self.cuml_params.copy()

        n_cols = self.n_cols
        dtype = self.dtype
        data_max_ = self.data_max_
        data_min_ = self.data_min_

        def _construct() -> CumlT:
            """
            Returns the instance of MinMaxScaler which will be passed to _transform_internal
            to do the transform.
            """
            import cupy as cp
            from cuml.preprocessing import MinMaxScaler as CumlMinMaxScaler

            from spark_rapids_ml.utils import cudf_to_cuml_array

            mms = CumlMinMaxScaler(output_type="numpy", **cuml_alg_params)
            mms.n_cols = n_cols
            mms.dtype = np.dtype(dtype)

            # just fit on [data_min_, data_max_] to compute scale_
            mms.fit(cp.array([data_min_, data_max_], dtype=mms.dtype))
            return mms

        def _transform_internal(
            mms_object: CumlT, df: Union[pd.DataFrame, np.ndarray]
        ) -> pd.DataFrame:
            res = mms_object.transform(df)
            return pd.Series(list(res))

        return _construct, _transform_internal, None

    def _out_schema(self, input_schema: Optional[StructType]) -> Union[StructType, str]:
        assert self.dtype is not None

        pyspark_type = dtype_to_pyspark_type(self.dtype)
        return f"array<{pyspark_type}>"
