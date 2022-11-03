#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

from typing import Any, Callable, Dict, List, Union

import cudf
import pandas as pd
from pyspark import Row, TaskContext
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from sparkcuml.core import (
    INIT_PARAMETERS_NAME,
    _CumlEstimator,
    _CumlModel,
    _set_pyspark_cuml_cls_param_attrs,
)


class CumlDummy(object):
    """
    A dummy class to mimic a cuml python class
    """

    def __init__(self, a=1, b=2, c=3) -> None:  # type: ignore
        super().__init__()
        self.a = a
        self.b = b
        self.c = c


class SparkCumlDummyModel(_CumlModel):
    """
    PySpark model of CumlDummy
    """

    def __init__(self, fake_model: int) -> None:
        super().__init__()
        self.fake_model = fake_model

    @classmethod
    def from_row(cls, row: Row) -> "SparkCumlDummyModel":
        return cls(row.dummy)

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Callable[[cudf.DataFrame], pd.DataFrame]:
        fake_model = self.fake_model

        def _dummy_transform(gdf: cudf.DataFrame) -> pd.DataFrame:
            assert fake_model == 1024
            return gdf.to_pandas()

        return _dummy_transform

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        return input_schema


class SparkCumlDummy(_CumlEstimator):
    """
    PySpark estimator of CumlDummy
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[List[cudf.DataFrame], Dict[str, Any]], Dict[str, Any]]:
        def _cuml_fit(
            df: List[cudf.DataFrame], params: Dict[str, Any]
        ) -> Dict[str, Any]:
            context = TaskContext.get()
            assert context is not None
            assert params["rank"] == context.partitionId()
            assert "handle" in params
            assert INIT_PARAMETERS_NAME in params
            init_params = params[INIT_PARAMETERS_NAME]
            assert init_params["a"] == 100
            assert "b" not in init_params
            assert init_params["c"] == 3
            dummy = CumlDummy(**init_params)
            assert dummy.a == 100
            assert dummy.b == 2
            assert dummy.c == 3

            import time

            # sleep for 1 sec to bypass https://issues.apache.org/jira/browse/SPARK-40932
            time.sleep(1)

            return {"dummy": [1024]}

        return _cuml_fit

    def _out_schema(self) -> Union[StructType, str]:
        return "dummy int"

    def _create_pyspark_model(self, result: Row) -> "SparkCumlDummyModel":
        assert result.dummy == 1024
        return SparkCumlDummyModel.from_row(result)

    @classmethod
    def _cuml_cls(cls) -> type:
        return CumlDummy

    @classmethod
    def _not_supported_param(cls) -> List[str]:
        """
        For some reason, spark cuml may not support all the parameters.
        In that case, we need to explicitly exclude them.
        """
        return ["b"]


_set_pyspark_cuml_cls_param_attrs(SparkCumlDummy, SparkCumlDummyModel)


def test_dummy(spark: SparkSession, gpu_number: int, tmp_path: str) -> None:
    data = [
        [1.0, 4.0, 4.0, 4.0],
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 2.0],
        [3.0, 3.0, 3.0, 2.0],
    ]

    rdd = spark.sparkContext.parallelize(data)
    input_cols = ["c1", "c2", "c3", "c4"]
    df = rdd.toDF(input_cols)
    df.show()

    dummy = SparkCumlDummy(inputCols=input_cols, a=100, num_workers=gpu_number)

    def assert_parameters(dummy: SparkCumlDummy) -> None:
        assert dummy.getInputCols() == input_cols
        assert dummy.getOrDefault(dummy.a) == 100  # type: ignore
        assert not dummy.hasParam("b")
        assert dummy.getOrDefault(dummy.c) == 3  # type: ignore

    assert_parameters(dummy)

    path = tmp_path + "/dummy_tests"
    estimator_path = f"{path}/dummy_estimator"
    dummy.write().overwrite().save(estimator_path)
    dummp_loaded = SparkCumlDummy.load(estimator_path)

    assert_parameters(dummp_loaded)

    model = dummy.fit(df)
    transformed_df = model.transform(df)
    ret = transformed_df.collect()
    assert len(ret) == 4

    for x, y in zip(ret, data):
        for i in range(len(ret)):
            assert x[i] == y[i]
