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

from typing import Tuple, Type, TypeVar

import pytest
from pyspark.ml.feature import MinMaxScaler as SparkMinMaxScaler
from pyspark.ml.feature import MinMaxScalerModel as SparkMinMaxScalerModel
from pyspark.ml.linalg import Vectors

from spark_rapids_ml.feature import MinMaxScaler, MinMaxScalerModel

from .sparksession import CleanSparkSession
from .utils import array_equal, assert_params

MinMaxScalerType = TypeVar(
    "MinMaxScalerType", Type[SparkMinMaxScaler], Type[MinMaxScaler]
)
MinMaxScalerModelType = TypeVar(
    "MinMaxScalerModelType", Type[SparkMinMaxScalerModel], Type[MinMaxScalerModel]
)


def test_params(gpu_number: int, tmp_path: str) -> None:
    # Default constructor
    default_spark_params = {
        "min": 0.0,
        "max": 1.0,
    }
    default_cuml_params = {
        "feature_range": (0.0, 1.0),
        "copy": True,
    }
    default_mms = MinMaxScaler()
    assert_params(default_mms, default_spark_params, default_cuml_params)

    # TODO
    # # Spark Params constructor
    # spark_params = {"min": -1.0, "max": 2.0}
    # spark_mms = MinMaxScaler(**spark_params)
    # expected_spark_params = default_spark_params.copy()
    # expected_spark_params.update(spark_params)  # type: ignore
    # expected_cuml_params = default_cuml_params.copy()
    # expected_cuml_params.update({"feature_range": (-1.0, 2.0)})
    # assert_params(spark_mms, expected_spark_params, expected_cuml_params)

    # # cuml_params constructor
    # cuml_params = {"feature_range": (-1.0, 2.0), "copy": False}
    # cuml_mms = MinMaxScaler(**cuml_params)
    # expected_spark_params = default_spark_params.copy()
    # expected_spark_params.update({"min": -1.0, "max": 2.0)})  # type: ignore
    # expected_cuml_params = default_cuml_params.copy()
    # expected_cuml_params.update(cuml_params)  # type: ignore
    # assert_params(cuml_mms, expected_spark_params, expected_cuml_params)

    # # Estimator persistence
    # path = tmp_path + "/mms_tests"
    # estimator_path = f"{path}/mms"
    # cuml_mms.write().overwrite().save(estimator_path)
    # custom_mms_loaded = MinMaxScaler.load(estimator_path)
    # assert_params(custom_mms_loaded, expected_spark_params, expected_cuml_params)

    # # Conflicting params
    # conflicting_params = {
    #     "min": 0.0,
    #     "feature_range": (-1.0, 2.0),
    # }
    # with pytest.raises(ValueError, match="set one or the other"):
    #     conflicting_mms = MinMaxScaler(**conflicting_params)


@pytest.mark.compat
@pytest.mark.parametrize(
    "pca_types",
    [(SparkMinMaxScaler, SparkMinMaxScalerModel), (MinMaxScaler, MinMaxScalerModel)],
)
def test_spark_compat(
    pca_types: Tuple[MinMaxScalerType, MinMaxScalerModelType],
    tmp_path: str,
) -> None:
    _MinMaxScaler, _MinMaxScalerModel = pca_types

    with CleanSparkSession() as spark:
        data = [
            (Vectors.dense([0.0, 10.0]),),
            (Vectors.dense([2.0, 5.0]),),
            (Vectors.dense([4.0, 7.5]),),
        ]
        df = spark.createDataFrame(data, ["input"])

        mms = _MinMaxScaler().setInputCol("input").setOutputCol("scaled")
        assert mms.getInputCol() == "input"
        assert mms.getOutputCol() == "scaled"

        model = mms.fit(df)
        assert model.getOutputCol() == "scaled"

        output = model.transform(df).collect()

        assert array_equal(output[0].scaled, [0.0, 1.0])
        assert array_equal(output[1].scaled, [0.5, 0.0])
        assert array_equal(output[2].scaled, [1.0, 0.5])

        expectedOriginalMin = Vectors.dense([0.0, 5.0])
        expectedOriginalMax = Vectors.dense([4.0, 10.0])
        assert model.originalMin == expectedOriginalMin
        assert model.originalMax == expectedOriginalMax

        mmsPath = tmp_path + "/mms"
        mms.write().overwrite().save(mmsPath)
        loaded_mms = _MinMaxScaler.load(mmsPath)
        assert loaded_mms.min == mms.min
        assert loaded_mms.max == mms.max

        modelPath = tmp_path + "/mms-model"
        model.write().overwrite().save(modelPath)
        loaded_model = _MinMaxScalerModel.load(modelPath)
        assert loaded_model.originalMin == model.originalMin
        assert loaded_model.originalMax == model.originalMax
        assert loaded_model.transform(df).take(1) == model.transform(df).take(1)
