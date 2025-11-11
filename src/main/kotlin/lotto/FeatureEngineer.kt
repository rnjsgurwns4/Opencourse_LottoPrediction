// src/main/kotlin/lotto/FeatureEngineer.kt
package lotto

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import org.jetbrains.kotlinx.dataframe.api.*


/**
 * 1. 과거 데이터를 ML 학습용 데이터셋(DataFrame)으로 변환
 * 2. 현재 시점의 예측용 특성을 생성
 */
class FeatureEngineer {

    // 학습 데이터를 생성할 때 필요한 최소 과거 데이터 수
    private val MIN_HISTORY = 10

    /**
     * ML 모델 훈련을 위한 전체 학습 데이터셋(X, Y)을 생성합니다.
     * @param history 훈련에 사용할 과거 데이터
     * @return 특성(X)과 정답(Y)이 포함된 DataFrame
     */
    fun createTrainingData(history: List<LottoTicket>): DataFrame<*> {
        val features = mutableListOf<Map<String, Any>>()

        // 11회차부터 마지막 회차까지 (10회차까지는 '특성' 계산용)
        for (i in MIN_HISTORY until history.size) {
            val currentDraw = history[i] // 현재 회차 (정답, Y)
            val pastDraws = history.subList(i - MIN_HISTORY, i) // 직전 10회차 (재료, X)
            val totalPastDraws = history.subList(0, i) // 1회차부터 직전까지 (누적 통계용)

            for (num in 1..45) {
                // 이 'num'이 'currentDraw'에 포함되었는지가 정답(Y)
                val label = currentDraw.numbers.contains(num)

                // 'num'에 대한 특성(X)을 맵으로 생성
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

        // 3. 'dataFrameOf()' 함수를 사용해,
        //    '열 이름'과 '데이터 리스트'를 짝지어 DataFrame을 재조립
        val df = dataFrameOf(
            "number" to numbers,
            "recency" to recencies,
            "freq_latest" to freqLats,
            "freq_total" to freqTots,
            "label" to labels
        )

        // --- ★★★ 사용자가 요청한 출력 코드 ★★★ ---

        // 2. DataFrame의 스키마(열 이름과 타입) 출력
        println("\n--- [DEBUG] DataFrame 스키마 (열 구조) ---")
        println(df.schema())
        println()

        // 3. DataFrame의 앞 10줄 내용 출력
        println("--- [DEBUG] DataFrame 내용 (앞 10줄) ---")
        println(df.take(10)) // .take(10)가 앞 10줄만 잘라서 보여줍니다.
        println("--------------------------------------------\n")

        return df
    }

    /**
     * '다음 회차'를 예측하기 위해, 현재 시점의 특성을 생성합니다.
     * @param fullHistory 1회차부터 현재까지의 모든 데이터 (누적 통계용)
     * @param latestDraws 최근 N회차 데이터 (최근 빈도, 미출현 기간용)
     * @return 1~45번 각 번호의 현재 특성 맵
     */
    fun createCurrentFeaturesForPrediction(
        fullHistory: List<LottoTicket>,
        latestDraws: List<LottoTicket>
    ): Map<Int, Map<String, Any>> {

        return (1..45).associateWith { num ->
            // 예측 시점에는 정답(label)이 없으므로 null
            createFeaturesForNumber(num, fullHistory, latestDraws, null)
        }
    }

    /**
     * 특정 번호(num)에 대한 특성 맵을 생성하는 핵심 로직
     */
    private fun createFeaturesForNumber(
        num: Int,
        totalPastDraws: List<LottoTicket>,
        latestDraws: List<LottoTicket>,
        label: Boolean? // 훈련 시에는 true/false, 예측 시에는 null
    ): Map<String, Any> {

        // 1. (Recency) 마지막으로 나온 지 얼마나 됐나?
        val recency = latestDraws.reversed().indexOfFirst { it.numbers.contains(num) }

        // 2. (Frequency) 최근 N회간 몇 번 나왔나?
        val freqLatest = latestDraws.count { it.numbers.contains(num) }

        // 3. (Total Frequency) 현재까지 총 몇 번 나왔나?
        val freqTotal = totalPastDraws.count { it.numbers.contains(num) }

        val featureMap = mutableMapOf<String, Any>()
        featureMap["number"] = num
        featureMap["recency"] = if (recency == -1) latestDraws.size else recency // -1이면 'N'주간 안 나옴
        featureMap["freq_latest"] = freqLatest
        featureMap["freq_total"] = freqTotal

        if (label != null) {
            // Weka는 boolean보다 문자열 'true', 'false'를 선호
            featureMap["label"] = label.toString()
        }

        return featureMap
    }
}