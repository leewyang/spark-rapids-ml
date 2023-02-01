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
import argparse
import datetime
import sys
from abc import abstractmethod
from distutils.util import strtobool
from typing import List, Any

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.sql import SparkSession

from benchmark.utils import with_benchmark, WithSparkSession


def _to_bool(literal: str) -> bool:
    return bool(strtobool(literal))


class BenchmarkBase:

    def __init__(self) -> None:
        """Add common parameters"""
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument("--gpu_workers", type=int, default=0,
                                  help="if gpu_workers > 0, How many gpu workers to be used for SparkCuml, "
                                       "else do the benchmark on Spark ML library ")
        self._parser.add_argument("--train_path", action="append", default=[], required=True,
                                  help="Input parquet format data path used for training")
        self._parser.add_argument("--transform_path", action="append", default=[],
                                  help="Input parquet format data path used for transform")
        self._parser.add_argument("--spark_confs", action="append", default=[])
        self._parser.add_argument("--no_shutdown", action='store_true', 
                                  help="do not stop spark session when finished")
        self.args_ = None

    def _parse_arguments(self, argv: List[Any]) -> None:
        self.args_ = self._parser.parse_args(argv)

    @property
    def args(self):
        return self.args_

    @abstractmethod
    def run(self, spark: SparkSession) -> None:
        raise NotImplementedError()


class BenchmarkLinearRegression(BenchmarkBase):

    def __init__(self, argv: List[Any]) -> None:
        super().__init__()
        self._parse_arguments(argv)

    def run(self, spark: SparkSession) -> None:

        df = spark.read.parquet(*self.args.train_path)

        label_name = "label"   
        is_array_col = True if any(['array' in t[1] for t in df.dtypes]) else False
        is_vector_col = True if any(['vector' in t[1] for t in df.dtypes]) else False
        is_single_col = is_array_col or is_vector_col


        features_col = [c for c in df.schema.names if c != label_name]
        if is_single_col:
            features_col = features_col[0]

        if self.args.gpu_workers > 0:
            from spark_rapids_ml.regression import LinearRegression
            lr = LinearRegression(num_workers=self.args.gpu_workers, verbose=7)
            lr.setFeaturesCol(features_col)
            lr.setLabelCol(label_name)
            model = with_benchmark("SparkCuml LinearRegression training:", lambda: lr.fit(df))

        else:
            from pyspark.ml.regression import LinearRegression
            spark_lr = LinearRegression()
            spark_lr.setLabelCol(label_name)
            spark_lr.setFeaturesCol("features")

            train_df = (
                df.select(array_to_vector(features_col).alias("features"), label_name)
            ) if is_array_col else (
                VectorAssembler()
                .setInputCols(features_col)
                .setOutputCol("features")
                .transform(df)
                .select("features", label_name)
            ) if not is_vector_col else df

            model = with_benchmark("Spark ML LinearRegression training:", lambda: spark_lr.fit(train_df))


class BenchmarkRunner:
    def __init__(self):
        registered_algorithms = {
            "linear_regression": BenchmarkLinearRegression
        }

        parser = argparse.ArgumentParser(
            description='Benchmark Spark CUML',
            usage='''benchmark_runner.py <algorithm> [<args>]

        Supported algorithms are:
           linear_regression     Benchmark linear regression
        ''')
        parser.add_argument('algorithm', help='benchmark the ML algorithms')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])

        if args.algorithm not in registered_algorithms:
            print("Unrecognized algorithm: ", args.algorithm)
            parser.print_help()
            exit(1)

        self._runner: BenchmarkBase = registered_algorithms[args.algorithm](sys.argv[2:])

    def run(self) -> None:
        args = self._runner.args
        with WithSparkSession(args.spark_confs, shutdown=(not args.no_shutdown)) as spark:
            with_benchmark("Total running time: ", lambda: self._runner.run(spark))


if __name__ == "__main__":
    """
    There're two ways to do the benchmark.

    1.
        python benchmark_runner.py [linear_regression] \
            --gpu_workers=2 \
            --train_path=xxx \
            --spark_confs="spark.master=local[12]" \

    2.
        spark-submit --master local[12] benchmark_runner.py -gpu_workers=2 --train_path=xxx
    """

    print(f"invoked time: {datetime.datetime.now()}")
    
    BenchmarkRunner().run()
