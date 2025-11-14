package lotto

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.*



class FeatureEngineer {

    private val SHORT_TERM_HISTORY = 10
    private val MID_TERM_HISTORY = 25

    // 학습 데이터셋 생성
    fun createTrainingData(history: List<LottoTicket>): DataFrame<*> {
        val features = mutableListOf<Map<String, Any>>()

        // 25회차부터 마지막 회차까지
        for (i in MID_TERM_HISTORY until history.size) {
            val currentDraw = history[i] // 현재 회차
            val pastDrawsShort = history.subList(i -  SHORT_TERM_HISTORY, i) // 직전 10회차
            val pastDrawsMid = history.subList(i - MID_TERM_HISTORY, i)
            val totalPastDraws = history.subList(0, i) // 1회차부터 직전까지 (누적 통계용)

            for (num in 1..45) {
                // num이 currentDraw에 포함되었는지
                val label = currentDraw.numbers.contains(num)

                // num에 대한 특성을 맵으로 생성
                features.add(
                    createFeaturesForNumber(num, totalPastDraws, pastDrawsShort, pastDrawsMid, label)
                )
            }
        }

        val numbers = features.map { it["number"] as Int }
        val recencies = features.map { it["recency"] as Int }
        val freqShorts = features.map { it["freq_short"] as Int }
        val freqMids = features.map { it["freq_mid"] as Int }
        val freqMain = features.map { it["freq_total_main"] as Int }
        val freqBonus = features.map { it["freq_total_bonus"] as Int }
        val labels = features.map { it["label"] as String }

        val df = dataFrameOf(
            "number" to numbers,
            "recency" to recencies,
            "freq_short" to freqShorts,
            "freq_mid" to freqMids,
            "freq_total_main" to freqMain,
            "freq_total_bonus" to freqBonus,
            "label" to labels
        )

        return df
    }

    // 모든 회차의 특성 생성
    fun createCurrentFeaturesForPrediction(
        fullHistory: List<LottoTicket>,
        latestDrawsShort: List<LottoTicket>,
        latestDrawsMid: List<LottoTicket>
    ): Map<Int, Map<String, Any>> {

        return (1..45).associateWith { num ->
            createFeaturesForNumber(num, fullHistory, latestDrawsShort, latestDrawsMid, null)
        }
    }

    //1 ~ 45의 특성 생성
    private fun createFeaturesForNumber(
        num: Int,
        totalPastDraws: List<LottoTicket>,
        latestDrawsShort: List<LottoTicket>,
        latestDrawsMid: List<LottoTicket>,
        label: Boolean? // 훈련 시에는 true/false, 예측 시에는 null
    ): Map<String, Any> {

        // 마지막으로 나온지 얼마나 됐는지
        val recency = latestDrawsMid.reversed().indexOfFirst { it.numbers.contains(num) }

        // 최근 N회간 나온 횟수
        val freqShort = latestDrawsShort.count { it.numbers.contains(num) }
        val freqMid = latestDrawsMid.count { it.numbers.contains(num) }

        // 현재까지 나온 횟수
        val freqTotalMain = totalPastDraws.count { it.numbers.contains(num) }
        val freqTotalBonus = totalPastDraws.count { it.bonusNo == num }

        val featureMap = mutableMapOf<String, Any>()
        featureMap["number"] = num
        featureMap["recency"] = if (recency == -1) latestDrawsMid.size else recency // -1이면 N주간 안 나옴
        featureMap["freq_short"] = freqShort
        featureMap["freq_mid"] = freqMid
        featureMap["freq_total_main"] = freqTotalMain
        featureMap["freq_total_bonus"] = freqTotalBonus

        if (label != null) {
            featureMap["label"] = label.toString()
        }

        return featureMap
    }
}