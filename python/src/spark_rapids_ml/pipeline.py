from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import cudf
import cupy
import numpy as np
import pandas as pd
from pyspark import RDD
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from spark_rapids_ml.common.cuml_context import CumlContext
from spark_rapids_ml.core import FitInputType
from spark_rapids_ml.utils import _get_spark_session, _is_local, get_logger

from .core import (
    CumlT,
    GPUEstimator,
    GPUModel,
    GPUTransformer,
    ParamAlias,
    _ConstructFunc,
    _CumlCommon,
    _CumlEstimator,
    _CumlModel,
    _CumlTransformer,
    _EvaluateFunc,
    _TransformFunc,
    alias,
    param_alias,
    transform_evaluate,
)

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap


class _CumlPipeline(_CumlEstimator):
    def __init__(self, stages: List[Union[GPUEstimator, GPUTransformer]]):
        self.stages = stages

    # def _fit(self, df: DataFrame) -> "_CumlPipelineModel":
    #     gdf = cudf.from_pandas(df)
    #     indexOfLastEstimator = -1
    #     for i, stage in enumerate(self.stages):
    #         if isinstance(stage, _CumlEstimator):
    #             indexOfLastEstimator = i

    #     transformers: List[_CumlTransformer] = []
    #     for i, stage in enumerate(self.stages):
    #         if i <= indexOfLastEstimator:
    #             if isinstance(stage, _CumlTransformer):
    #                 transformers.append(stage)
    #                 gdf = stage.gpu_transform(gdf)
    #             elif isinstance(stage, _CumlEstimator):
    #                 model = stage.gpu_fit(gdf)
    #                 model_transformer = cast(_CumlTransformer, model)
    #                 transformers.append(model_transformer)
    #                 if i < indexOfLastEstimator:
    #                     gdf = model_transformer.gpu_transform(gdf)
    #             else:
    #                 raise ValueError(
    #                     f"Pipeline stage is not a _CumlEstimator or _CumlTransformer: {stage}"
    #                 )
    #         else:
    #             transformers.append(cast(_CumlTransformer, stage))

    #     return _CumlPipelineModel(transformers)

    def _get_cuml_fit_func(
        self, dataset: DataFrame, extra_params: Optional[List[Dict[str, Any]]] = None
    ) -> Callable[[FitInputType, Dict[str, Any]], Dict[str, Any]]:
        def _cuml_fit(dataset: FitInputType, params: Dict[str, Any]) -> Dict[str, Any]:
            # TODO: support cp.ndarray
            assert isinstance(dataset, cudf.DataFrame)
            gdf: cudf.DataFrame = dataset

            indexOfLastEstimator = -1
            for i, stage in enumerate(self.stages):
                if isinstance(stage, _CumlEstimator):
                    indexOfLastEstimator = i

            transformers: List[_CumlTransformer] = []
            for i, stage in enumerate(self.stages):
                if i <= indexOfLastEstimator:
                    if isinstance(stage, _CumlTransformer):
                        transformers.append(stage)
                        gdf = stage.gpu_transform(gdf)
                    elif isinstance(stage, _CumlEstimator):
                        model = stage.gpu_fit(gdf)
                        model_transformer = cast(_CumlTransformer, model)
                        transformers.append(model_transformer)
                        if i < indexOfLastEstimator:
                            gdf = model_transformer.gpu_transform(gdf)
                    else:
                        raise ValueError(
                            f"Pipeline stage is not a _CumlEstimator or _CumlTransformer: {stage}"
                        )
                else:
                    transformers.append(cast(_CumlTransformer, stage))

            # return _CumlPipelineModel(transformers)
            return {}

        return _cuml_fit


class _CumlPipelineModel(_CumlModel):
    def __init__(self, stages: List[_CumlTransformer]):
        self.stages = stages

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {}

    def _get_cuml_transform_func(
        self, category: str = transform_evaluate.transform
    ) -> Tuple[_ConstructFunc, _TransformFunc, Optional[_EvaluateFunc]]:
        stages = self.stages

        def construct_stages() -> CumlT:
            cumlObjs: List[CumlT] = []
            for stage in stages:
                make_fn, _, _ = stage._get_cuml_transform_func(category=category)
                cumlObjs.append(make_fn())
            return cumlObjs

        def transform_stages(cuml_obj: CumlT, gdf: cudf.DataFrame) -> cudf.DataFrame:
            for i, stage in enumerate(stages):
                _, transform_fn, _ = stage._get_cuml_transform_func(category=category)
                gdf = transform_fn(cuml_obj[i], gdf)
            return gdf

        return construct_stages, transform_stages, None

    def _out_schema(
        self, input_schema: Optional[Union[StructType, str]]
    ) -> Union[StructType, str]:
        # output schema of the last transform stage
        last_stage = cast(_CumlTransformer, self.stages[-1])
        return last_stage._out_schema(None)
