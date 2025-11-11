// src/main/kotlin/lotto/LottoPredictor.kt
package lotto

import weka.classifiers.Classifier
import weka.core.DenseInstance
import weka.core.Instance
import weka.core.Instances

// 예측
class LottoPredictor(
    private val models: Map<Int, Classifier>, // 45개 모델
    private val dataHeader: Instances,     // 훈련 시 사용한 데이터 헤더
    private val featureEngineer: FeatureEngineer
) {

    // Instance 객체로 변환
    private fun convertFeaturesToInstance(features: Map<String, Any>): Instance {
        val instance = DenseInstance(dataHeader.numAttributes())
        instance.setDataset(dataHeader)

        instance.setValue(dataHeader.attribute("recency"), (features["recency"] as Int).toDouble())
        instance.setValue(dataHeader.attribute("freq_latest"), (features["freq_latest"] as Int).toDouble())
        instance.setValue(dataHeader.attribute("freq_total"), (features["freq_total"] as Int).toDouble())

        return instance
    }

    fun predictNextDraw(
        fullHistory: List<LottoTicket>,
        latestDraws: List<LottoTicket>
    ): List<Int> {
        println("[LottoPredictor] '다음 회차' 예측을 위한 특성 계산 시작...")
        // 1. '현재 시점'의 특성을 1회 계산
        val currentFeaturesMap = featureEngineer.createCurrentFeaturesForPrediction(fullHistory, latestDraws)

        // 2. 미리 계산된 특성으로 예측 실행
        return predictFromFeatures(currentFeaturesMap)
    }

    fun predictFromFeatures(
        currentFeaturesMap: Map<Int, Map<String, Any>>
    ): List<Int> {
        println("[LottoPredictor] 45개 모델을 순회하며 '당첨 확률' 계산 중...")
        val probabilities = mutableListOf<Pair<Int, Double>>()

        for (num in 1..45) {
            val model = models[num] ?: continue
            val features = currentFeaturesMap[num] ?: continue

            val currentInstance = convertFeaturesToInstance(features)
            val distribution = model.distributionForInstance(currentInstance)

            val trueIndex = dataHeader.attribute("label").indexOfValue("true")
            val probability = distribution[trueIndex]
            probabilities.add(Pair(num, probability))
        }

        println("[LottoPredictor] 확률 계산 완료. 상위 6개 번호 추출 중...")

        val top6 = probabilities
            .sortedByDescending { it.second }
            .take(6)

        println("--- 예측 확률 상위 6개 ---")
        top6.forEachIndexed { index, pair ->
            println("${index + 1}순위: ${pair.first}번 (예측 확률: ${"%.4f".format(pair.second)})")
        }

        return top6.map { it.first }.sorted()
    }

    // 예측
    fun predictTop6(
        fullHistory: List<LottoTicket>,
        latestDraws: List<LottoTicket>
    ): List<Int> {

        println("다음 회차 예측을 시작합니다...")

        // 현재 시점의 특성 맵 생성 (1~45번)
        val currentFeaturesMap = featureEngineer.createCurrentFeaturesForPrediction(fullHistory, latestDraws)

        val probabilities = mutableListOf<Pair<Int, Double>>()

        // 모델 실행
        for (num in 1..45) {
            val model = models[num] ?: continue
            val features = currentFeaturesMap[num] ?: continue

            // 현재 특성을 Weka Instance로 변환
            val currentInstance = convertFeaturesToInstance(features)

            val distribution = model.distributionForInstance(currentInstance)

            //확률 생성
            val trueIndex = dataHeader.attribute("label").indexOfValue("true")
            val probability = distribution[trueIndex]

            probabilities.add(Pair(num, probability))
        }

        return probabilities
            .sortedByDescending { it.second }
            .take(6)
            .map { it.first }
            .sorted()
    }
}