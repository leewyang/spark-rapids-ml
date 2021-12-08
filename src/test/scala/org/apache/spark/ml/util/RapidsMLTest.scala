/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.util

import org.apache.spark.SparkConf
import org.apache.spark.sql.DataFrame

trait RapidsMLTest extends MLTest {
  override def sparkConf: SparkConf = {
    super.sparkConf.set("spark.rapids.sql.enabled", "true")
  }

  override def checkVectorSizeOnDF(
      dataframe: DataFrame,
      vecColName: String,
      vecSize: Int): Unit = {
    super.checkVectorSizeOnDF(dataframe, vecColName, vecSize)
  }
}
