package lotto

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.*


class FeatureEngineer {

    private val MIN_HISTORY = 10

    // 학습 데이터셋 생성
    fun createTrainingData(history: List<LottoTicket>): DataFrame<*> {
        val features = mutableListOf<Map<String, Any>>()

        // 11회차부터 마지막 회차까지
        for (i in MIN_HISTORY until history.size) {
            val currentDraw = history[i] // 현재 회차
            val pastDraws = history.subList(i - MIN_HISTORY, i) // 직전 10회차
            val totalPastDraws = history.subList(0, i) // 1회차부터 직전까지 (누적 통계용)

            for (num in 1..45) {
                // num이 currentDraw에 포함되었는지
                val label = currentDraw.numbers.contains(num)

                // num에 대한 특성을 맵으로 생성
                features.add(
                    createFeaturesForNumber(num, totalPastDraws, pastDraws, label)
                )
            }
        }

        val numbers = features.map { it["number"] as Int }
        val recencies = features.map { it["recency"] as Int }
        val freqLats = features.map { it["freq_latest"] as Int }
        val freqTots = features.map { it["freq_total"] as Int }
        val labels = features.map { it["label"] as String }

        val df = dataFrameOf(
            "number" to numbers,
            "recency" to recencies,
            "freq_latest" to freqLats,
            "freq_total" to freqTots,
            "label" to labels
        )

        return df
    }

    // 모든 회차의 특성 생성
    fun createCurrentFeaturesForPrediction(
        fullHistory: List<LottoTicket>,
        latestDraws: List<LottoTicket>
    ): Map<Int, Map<String, Any>> {

        return (1..45).associateWith { num ->
            createFeaturesForNumber(num, fullHistory, latestDraws, null)
        }
    }

    //1 ~ 45의 특성 생성
    private fun createFeaturesForNumber(
        num: Int,
        totalPastDraws: List<LottoTicket>,
        latestDraws: List<LottoTicket>,
        label: Boolean? // 훈련 시에는 true/false, 예측 시에는 null
    ): Map<String, Any> {

        // 마지막으로 나온지 얼마나 됐는지
        val recency = latestDraws.reversed().indexOfFirst { it.numbers.contains(num) }

        // 최근 N회간 나온 횟수
        val freqLatest = latestDraws.count { it.numbers.contains(num) }

        // 현재까지 나온 횟수
        val freqTotal = totalPastDraws.count { it.numbers.contains(num) }

        val featureMap = mutableMapOf<String, Any>()
        featureMap["number"] = num
        featureMap["recency"] = if (recency == -1) latestDraws.size else recency // -1이면 N주간 안 나옴
        featureMap["freq_latest"] = freqLatest
        featureMap["freq_total"] = freqTotal

        if (label != null) {
            featureMap["label"] = label.toString()
        }

        return featureMap
    }
}