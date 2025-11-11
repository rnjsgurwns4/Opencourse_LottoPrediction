// src/main/kotlin/lotto/LottoPredictor.kt
package lotto

import weka.classifiers.Classifier
import weka.core.DenseInstance
import weka.core.Instance
import weka.core.Instances

/**
 * 훈련된 45개의 모델을 사용해 최종 6개 번호를 예측
 */
class LottoPredictor(
    private val models: Map<Int, Classifier>, // 45개 모델
    private val dataHeader: Instances,     // 훈련 시 사용한 데이터 헤더(구조)
    private val featureEngineer: FeatureEngineer
) {

    /**
     * 현재 특성(Map)을 Weka의 Instance 객체로 변환
     */
    private fun convertFeaturesToInstance(features: Map<String, Any>): Instance {
        val instance = DenseInstance(dataHeader.numAttributes())
        instance.setDataset(dataHeader) // 훈련 때 쓴 헤더와 연결 (필수!)

        instance.setValue(dataHeader.attribute("recency"), (features["recency"] as Int).toDouble())
        instance.setValue(dataHeader.attribute("freq_latest"), (features["freq_latest"] as Int).toDouble())
        instance.setValue(dataHeader.attribute("freq_total"), (features["freq_total"] as Int).toDouble())
        // 'label'은 예측 시점에는 없으므로 비워둠 (Weka가 알아서 처리)

        return instance
    }

    /**
     * 다음 회차에 나올 확률이 가장 높은 6개 번호를 예측합니다.
     * @param fullHistory 1회차부터 현재까지 모든 데이터 (누적 통계용)
     * @param latestDraws 최근 N회차 데이터 (최근 특성용)
     */
    fun predictTop6(
        fullHistory: List<LottoTicket>,
        latestDraws: List<LottoTicket>
    ): List<Int> {

        println("다음 회차 예측을 시작합니다...")

        // 1. 현재 시점의 특성 맵 생성 (1~45번)
        val currentFeaturesMap = featureEngineer.createCurrentFeaturesForPrediction(fullHistory, latestDraws)

        val probabilities = mutableListOf<Pair<Int, Double>>()

        // 2. 1번부터 45번까지 각 모델에게 '당첨 확률'을 물어봄
        for (num in 1..45) {
            val model = models[num] ?: continue
            val features = currentFeaturesMap[num] ?: continue

            // 3. 현재 특성을 Weka Instance로 변환
            val currentInstance = convertFeaturesToInstance(features)

            // 4. 모델에게 '당첨(true)될 확률'을 물어봄
            // [0.8 (false 확률), 0.2 (true 확률)] 같은 배열 반환
            val distribution = model.distributionForInstance(currentInstance)

            // "true" 레이블의 인덱스를 찾아서 해당 확률을 가져옴 (보통 0)
            val trueIndex = dataHeader.attribute("label").indexOfValue("true")
            val probability = distribution[trueIndex]

            probabilities.add(Pair(num, probability))
        }

        // 5. 확률(probability) 기준 내림차순 정렬 후 상위 6개 번호(num) 추출
        return probabilities
            .sortedByDescending { it.second } // 확률(second)로 정렬
            .take(6)                           // 상위 6개
            .map { it.first }                  // 번호(first)만 반환
            .sorted()
    }
}