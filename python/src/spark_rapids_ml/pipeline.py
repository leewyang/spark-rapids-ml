from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import cudf
import json
from pyspark.ml.param.shared import HasInputCol
from pyspark.sql import Row
from pyspark.sql.types import Row, StructType, StructField, StringType
from spark_rapids_ml.core import _CumlModel, FitInputType
from .core import (
    CumlT,
    _ConstructFunc,
    _CumlEstimator,
    _CumlModel,
    _CumlTransformer,
    _EvaluateFunc,
    _TransformFunc,
    transform_evaluate,
)
from .params import P, _CumlParams


class _CumlPipeline(_CumlEstimator, _CumlParams, HasInputCol):
    def __init__(self, stages: List[Union[_CumlEstimator, _CumlTransformer]]):
        super().__init__()
        self.stages = stages

    def _get_input_columns(self) -> Tuple[Optional[str], Optional[List[str]]]:
        return self.stages[0]._get_input_columns()

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {}

    def _get_cuml_fit_func(
        self, extra_params: Optional[List[Dict[str, Any]]] = None
    ) -> Callable[[FitInputType, Dict[str, Any]], Dict[str, Any]]:
        def _cuml_fit(dataset: FitInputType, params: Dict[str, Any]) -> Dict[str, Any]:

            indexOfLastEstimator = -1
            for i, stage in enumerate(self.stages):
                if isinstance(stage, _CumlEstimator):
                    indexOfLastEstimator = i

            transformer_params = []
            for i, stage in enumerate(self.stages):
                if i <= indexOfLastEstimator:
                    if isinstance(stage, _CumlTransformer):
                        # invoke transformers on dataset
                        transformer_params.append({stage.__class__.__name__: stage.cuml_params})
                        construct_fn, transform_fn, _ = model._get_cuml_transform_func()
                        cuml_obj = construct_fn()
                        dataset = transform_fn(cuml_obj, dataset)
                    elif isinstance(stage, _CumlEstimator):
                        # fit all estimators to convert them to models/transformers
                        cuml_fit_func = stage._get_cuml_fit_func(params)
                        model_params = cuml_fit_func(dataset, params)
                        model = stage._create_pyspark_model(Row(**model_params))
                        transformer_params.append({model.__class__.__name__: model.get_model_attributes()})
                        if i < indexOfLastEstimator:
                            # for all models/transformers upstream of the last estimator/model
                            # invoke model.transform() on dataset
                            construct_fn, transform_fn, _ = model._get_cuml_transform_func()
                            cuml_obj = construct_fn()
                            dataset = transform_fn(cuml_obj, dataset)
                    else:
                        raise ValueError(
                            f"Pipeline stage is not a _CumlEstimator or _CumlTransformer: {stage}"
                        )
                else:
                    transformer_params.append({stage.__class__.__name__: stage.cuml_params})

            # represent stages as a single row with one JSON field called "stages"
            return {"stages": json.dumps(transformer_params)}

        return _cuml_fit

    # _create_pyspark_model, _out_schema
    def _create_pyspark_model(self, result: Row) -> _CumlModel:
        transformer_stages = result.asDict()
        # return super()._create_pyspark_model(result)

    def _out_schema(self) -> Union[StructType, str]:
        return StructType([StructField("stages", StringType(), True)])


class _CumlPipelineModel(_CumlModel):
    def __init__(self, stages: List[Dict[str, Any]]):
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
