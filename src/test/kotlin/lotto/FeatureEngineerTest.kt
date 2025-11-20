package lotto

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.jetbrains.kotlinx.dataframe.api.*

class FeatureEngineerTest {

    @Test
    fun `5가지 특성이 모두 정확히 계산되는지 검증`() {
        val mockHistory = mutableListOf<LottoTicket>()

        for (i in 1..50) {
            // 기본: 1, 10, 11, 12, 13, 14 / 보너스 3
            val numbers = mutableSetOf(1, 10, 11, 12, 13, 14)
            var bonus = 3

            // 시나리오 설정
            // 1번 공: 매회 당첨
            // 2번 공: 한 번도 안 나옴
            // 3번 공: 매회 보너스로만 나옴

            // 4번 공: Recency 테스트용
            // 45회차에 딱 한 번 나오고, 그 뒤로 안 나옴
            if (i == 45) {
                numbers.remove(14) // 자리 만들기
                numbers.add(4)
            }

            mockHistory.add(LottoTicket(i, numbers, bonus))
        }

        // FeatureEngineer 실행
        val featureEngineer = FeatureEngineer()
        val dataFrame = featureEngineer.createTrainingData(mockHistory)

        // 1번 공 (항상 나옴)
        val row1 = dataFrame.rows().last { (it["number"] as Int) == 1 }

        assertEquals(0, row1["recency"], "1번: 직전 회차(49회)에 나왔으므로 Recency는 0이어야 함")
        assertEquals(10, row1["freq_short"], "1번: 최근 10회 모두 나왔으므로 10이어야 함")
        assertEquals(25, row1["freq_mid"], "1번: 최근 25회 모두 나왔으므로 25이어야 함")
        assertEquals(49, row1["freq_total_main"], "1번: 1~49회 모두 나왔으므로 누적 49여야 함")
        assertEquals(0, row1["freq_total_bonus"], "1번: 보너스로는 안 나왔으므로 0이어야 함")

        // 2번 공 (아예 안 나옴)
        val row2 = dataFrame.rows().last { (it["number"] as Int) == 2 }

        assertEquals(25, row2["recency"], "2번: 안 나왔으므로 Recency는 WindowSize(25)여야 함")
        assertEquals(0, row2["freq_short"], "2번: 안 나왔으므로 단기 빈도 0")
        assertEquals(0, row2["freq_total_main"], "2번: 누적 빈도 0")

        // 3번 공 (항상 보너스)
        val row3 = dataFrame.rows().last { (it["number"] as Int) == 3 }

        assertEquals(0, row3["freq_total_main"], "3번: 메인 번호로는 안 나옴 (0)")
        assertEquals(49, row3["freq_total_bonus"], "3번: 매번 보너스로 나옴 (49)")

        // 4번 공 (Recency 테스트)
        // 1~49회차 데이터 기준.
        // 최근 순서: 49(X), 48(X), 47(X), 46(X), 45(O)
        val row4 = dataFrame.rows().last { (it["number"] as Int) == 4 }

        println("4번 공 Recency 값: ${row4["recency"]}")
        assertEquals(4, row4["recency"], "4번: 45회에 나오고 49회까지 안 나왔으므로(4칸) Recency는 4")

        println("\n모든 특성 테스트 통과")
    }
}