package lotto

enum class Rank {
    FIRST,  // 1등
    SECOND, // 2등
    THIRD,  // 3등
    FOURTH, // 4등
    FIFTH,  // 5등
    NONE;   // 꽝

    companion object {
        /**
         * 예측 번호와 실제 당첨 티켓을 비교하여 등수를 판별합니다.
         * @param predicted 예측한 6개 번호
         * @param actual 실제 당첨 정보 (보너스 번호 포함)
         */
        fun determineRank(predicted: Set<Int>, actual: LottoTicket): Rank {
            // 1. 예측 번호와 실제 당첨 번호(보너스 제외)가 몇 개 일치하는지
            val matchCount = predicted.intersect(actual.numbers).size

            // 2. 예측 번호에 보너스 번호가 포함되는지
            val bonusMatch = predicted.contains(actual.bonusNo)

            return when {
                matchCount == 6 -> FIRST
                matchCount == 5 && bonusMatch -> SECOND
                matchCount == 5 -> THIRD
                matchCount == 4 -> FOURTH
                matchCount == 3 -> FIFTH
                else -> NONE
            }
        }
    }
}