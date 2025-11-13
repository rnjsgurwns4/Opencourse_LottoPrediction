package lotto

import weka.classifiers.Classifier
import weka.core.DenseInstance
import weka.core.Instance
import weka.core.Instances
import kotlin.random.Random

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
        latestDraws: List<LottoTicket>,
        setsToGenerate: Int = 1
    ): List<List<Int>> {
        // 특성을 1회 계산
        val currentFeaturesMap = featureEngineer.createCurrentFeaturesForPrediction(fullHistory, latestDraws)

        // 미리 계산된 특성으로 예측 실행
        return predictFromFeatures(currentFeaturesMap, setsToGenerate)
    }

    fun predictFromFeatures(
        currentFeaturesMap: Map<Int, Map<String, Any>>,
        setsToGenerate: Int = 1
    ): List<List<Int>> {
        println("[LottoPredictor] 45개 모델을 순회하며 당첨 확률 계산 중")
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
        val finalPredictionSets = mutableListOf<List<Int>>()

        val top6Deterministic = probabilities
            .sortedByDescending { it.second }
            .take(6)
            .map { it.first }
            .sorted()

        // Top 6 조합을 리스트의 첫 번째로 추가
        finalPredictionSets.add(top6Deterministic)

        val probabilisticSetsToGenerate = setsToGenerate - 1

        if (probabilisticSetsToGenerate > 0) {

            val probabilisticSets = (1..probabilisticSetsToGenerate).map {
                getWeightedRandomSample(probabilities, 6)
            }
            // 확률 샘플링 결과를 리스트에 추가
            finalPredictionSets.addAll(probabilisticSets)
        }
        return finalPredictionSets
    }

    private fun getWeightedRandomSample(
        probabilities: List<Pair<Int, Double>>,
        k: Int = 6 // 6개 뽑기
    ): List<Int> {
        val picked = mutableListOf<Int>()
        val remaining = probabilities.toMutableList()

        repeat(k) {
            if (remaining.isEmpty()) return picked.sorted() // 뽑을 게 없으면 종료

            // 남아있는 번호들의 확률 총합
            val totalWeight = remaining.sumOf { it.second }
            if (totalWeight <= 0) return picked.sorted() // 확률 총합이 0이면 종료

            var randomTarget = Random.nextDouble() * totalWeight

            var selected: Pair<Int, Double>? = null
            for (pair in remaining) {
                randomTarget -= pair.second
                if (randomTarget <= 0) {
                    selected = pair
                    break
                }
            }

            // 부동소수점 오류 대비
            val numberToPick = selected ?: remaining.last()

            picked.add(numberToPick.first) // 뽑힌 리스트에 추가
            remaining.remove(numberToPick) // 뽑힌 번호는 풀에서 제거
        }
        return picked.sorted()
    }
}