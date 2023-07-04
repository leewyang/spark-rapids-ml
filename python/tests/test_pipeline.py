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

from typing import List, Tuple, Type, TypeVar

from .sparksession import CleanSparkSession

def test_spark_pipeline(gpu_number: int, tmp_path: str) -> None:
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import MinMaxScaler
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.pipeline import Pipeline

    data = [(Vectors.dense([0.0, 0.0]), 2.0), (Vectors.dense([1.0, 1.0]), 2.0),
            (Vectors.dense([9.0, 8.0]), 2.0), (Vectors.dense([8.0, 9.0]), 2.0)]

    # KMeans only
    with CleanSparkSession() as spark:
        df = spark.createDataFrame(data, ["features", "weighCol"])
        kmeans = KMeans().setK(2).setFeaturesCol("features").setPredictionCol("predictions")
        pipe = Pipeline(stages=[kmeans])
        model = pipe.fit(df)
        preds = model.transform(df)
        preds.show()

    # MinMaxScaler + KMeans
    with CleanSparkSession() as spark:
        df = spark.createDataFrame(data, ["features", "weighCol"])
        scaler = MinMaxScaler().setInputCol("features").setOutputCol("scaled_features")
        kmeans = KMeans().setFeaturesCol("scaled_features").setPredictionCol("predictions")
        pipe = Pipeline(stages=[scaler, kmeans])
        model = pipe.fit(df)
        preds = model.transform(df)
        preds.show()

def test_cuml_pipeline(gpu_number: int, tmp_path: str) -> None:
    from pyspark.ml.linalg import Vectors
    # from spark_rapids_ml.feature import MinMaxScaler
    from spark_rapids_ml.clustering import KMeans
    from spark_rapids_ml.pipeline import _CumlPipeline

    data = [(Vectors.dense([0.0, 0.0]), 2.0), (Vectors.dense([1.0, 1.0]), 2.0),
            (Vectors.dense([9.0, 8.0]), 2.0), (Vectors.dense([8.0, 9.0]), 2.0)]

    # KMeans only
    with CleanSparkSession() as spark:
        df = spark.createDataFrame(data, ["features", "weighCol"])
        kmeans = KMeans().setK(2).setFeaturesCol("features").setPredictionCol("predictions")

        pipe = _CumlPipeline(stages=[kmeans])
        model = pipe.fit(df)

        preds = model.transform(df)
        preds.show()

    # # MinMaxScaler + KMeans
    # with CleanSparkSession() as spark:
    #     df = spark.createDataFrame(data, ["features", "weighCol"])
    #     scaler = MinMaxScaler().setInputCol("features").setOutputCol("scaled_features")
    #     kmeans = KMeans().setK(2).setFeaturesCol("features").setPredictionCol("predictions")
    #     pipe = _CumlPipeline(stages=[scaler, kmeans])
    #     model = pipe.fit(df)
    #     preds = model.transform(df)
    #     preds.show()