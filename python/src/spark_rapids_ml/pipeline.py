import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from pyspark.ml.param.shared import (
    HasFeaturesCol,
    HasInputCol,
    HasInputCols,
    HasOutputCol,
    HasOutputCols,
    HasPredictionCol,
)
from pyspark.sql import DataFrame, Row
from pyspark.sql.types import IntegerType, Row, StringType, StructField, StructType

from .clustering import KMeansModel
from .core import (
    CumlT,
    FitInputType,
    _ConstructFunc,
    _CumlCommon,
    _CumlEstimator,
    _CumlModel,
    _EvaluateFunc,
    _TransformFunc,
    transform_evaluate,
)
from .feature import MinMaxScaler, MinMaxScalerModel
from .params import HasFeaturesCols, P, _CumlParams
from .utils import _concat_and_free, _get_gpu_id, _get_spark_session, _is_local


class _CumlPipelineParams(
    _CumlParams, HasFeaturesCol, HasFeaturesCols, HasPredictionCol
):
    """
    Shared Spark Params for KMeans and KMeansModel.
    """

    def __init__(self) -> None:
        super().__init__()

    def getFeaturesCol(self) -> Union[str, List[str]]:  # type: ignore
        """
        Gets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`
        """
        if self.isDefined(self.featuresCols):
            return self.getFeaturesCols()
        elif self.isDefined(self.featuresCol):
            return self.getOrDefault("featuresCol")
        else:
            raise RuntimeError("featuresCol is not set")

    def setFeaturesCol(self: P, value: Union[str, List[str]]) -> P:
        """
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`. Used when input vectors are stored in a single column.
        """
        if isinstance(value, str):
            self.set_params(featuresCol=value)
        else:
            self.set_params(featuresCols=value)
        return self

    def setFeaturesCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`featuresCols`. Used when input vectors are stored as multiple feature columns.
        """
        return self.set_params(featuresCols=value)

    def setPredictionCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        self.set_params(predictionCol=value)
        return self


class _CumlPipeline(_CumlEstimator, _CumlParams, HasInputCol):
    def __init__(self, stages: List[_CumlEstimator]):
        super().__init__()
        self.stages = stages

    def _get_input_columns(self) -> Tuple[Optional[str], Optional[List[str]]]:
        return self.stages[0]._get_input_columns()

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {}

    def _get_cuml_fit_func(
        self, extra_params: Optional[List[Dict[str, Any]]] = None
    ) -> Callable[[FitInputType, Dict[str, Any]], Dict[str, Any]]:
        def _cuml_fit(dfs: FitInputType, params: Dict[str, Any]) -> Dict[str, Any]:
            X_list = [item[0] for item in dfs]
            y_list = [item[1] for item in dfs]
            if isinstance(X_list[0], pd.DataFrame):
                X = pd.concat(X_list)
                y = pd.concat(y_list) if any(y_list) else None
            else:
                # features are either cp or np arrays here
                X = _concat_and_free(X_list)
                y = _concat_and_free(y_list) if any(y_list) else None

            indexOfLastEstimator = -1
            for i, stage in enumerate(self.stages):
                if isinstance(stage, _CumlEstimator):
                    indexOfLastEstimator = i

            transformer_params = []
            for i, stage in enumerate(self.stages):
                if i <= indexOfLastEstimator:
                    if isinstance(stage, _CumlEstimator):
                        # fit all estimators to convert them to models/transformers
                        cuml_fit_func = stage._get_cuml_fit_func(extra_params)
                        model_params = cuml_fit_func([(X, y, None)], params)
                        model = stage._create_pyspark_model(Row(**model_params))
                        transformer_params.append(
                            {
                                "class": model.__class__.__qualname__,
                                "params": model.get_model_attributes(),
                            }
                        )
                        if i < indexOfLastEstimator:
                            # for all models/transformers upstream of the last estimator/model
                            # invoke model.transform() on dataset
                            X = model._transform_evaluate_internal(
                                X, schema=model._out_schema(None)
                            )
                    else:
                        raise ValueError(
                            f"Pipeline stage is not a _CumlEstimator or _CumlModel: {stage}"
                        )
                else:
                    transformer_params.append(
                        {stage.__class__.__name__: stage.cuml_params}
                    )

            # TODO: find better representation?
            # represent stages as a single row with one JSON field called "stages"
            return {"stage_json": [json.dumps(transformer_params)]}

        return _cuml_fit

    def _create_pyspark_model(self, result: Row) -> _CumlModel:
        transformer_stages = result.asDict()
        model = _CumlPipelineModel.from_row(result)
        return model

    def _out_schema(self) -> Union[StructType, str]:
        return StructType([StructField("stage_json", StringType(), True)])


class _CumlPipelineModel(_CumlModel, HasInputCol, HasFeaturesCol, HasPredictionCol):
    def __init__(self, stage_json: List[Dict[str, Any]]):
        super().__init__()
        self.stage_json = stage_json

        # convert JSON into class instances
        stages: List[_CumlModel] = []
        for stage in self.stage_json:
            cls_name = stage["class"]
            cls_params = stage["params"]
            # TODO: dynamic instantiation based on fully qualified names
            if cls_name == "KMeansModel":
                stages.append(KMeansModel(**cls_params))
            elif cls_name == "MinMaxScalerModel":
                stages.append(MinMaxScalerModel(**cls_params))
        self.stages = stages

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_row(cls, result: Row) -> "_CumlPipelineModel":
        stage_json = json.loads(result.asDict()["stage_json"])
        return cls(stage_json=stage_json)

    def _get_cuml_transform_func(
        self, dataset: DataFrame, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc]]:
        return None, None, None  # type: ignore

    def _transform_evaluate_internal(
        self, dataset: DataFrame, schema: Union[StructType, str]
    ) -> DataFrame:
        """Internal API to support transform and evaluation in a single pass"""
        dataset, select_cols, input_is_multi_cols = self._pre_process_data(dataset)

        is_local = _is_local(_get_spark_session().sparkContext)

        # Get the functions which will be passed into executor to run.
        construct_functions: List[_ConstructFunc] = []
        transform_functions: List[_TransformFunc] = []
        for stage in self.stages:
            make_fn, transform_fn, _ = stage._get_cuml_transform_func(dataset)
            construct_functions.append(make_fn)
            transform_functions.append(transform_fn)

        array_order = self._transform_array_order()

        # TODO: get input and output column names

        def _transform_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            from pyspark import TaskContext

            context = TaskContext.get()

            _CumlCommon.set_gpu_device(context, is_local, True)

            # Construct the cuml counterpart objects
            cuml_objs = [construct_fn() for construct_fn in construct_functions]
            # transform_func = transform_func()

            for pdf in pdf_iter:
                for cuml_obj, transform_fn in zip(cuml_objs, transform_functions):
                    pdf = transform_fn(cuml_obj, pdf)
                yield pdf

        return dataset.mapInPandas(_transform_udf, schema=schema)  # type: ignore

    def _out_schema(self, input_schema: Optional[StructType]) -> Union[StructType, str]:
        # output schema of the last transform stage
        last_stage = cast(_CumlModel, self.stages[-1])
        return last_stage._out_schema(input_schema)
