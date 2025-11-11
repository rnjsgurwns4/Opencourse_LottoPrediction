package lotto

enum class Rank {
    FIRST,  // 1등
    SECOND, // 2등
    THIRD,  // 3등
    FOURTH, // 4등
    FIFTH,  // 5등
    NONE;   // 꽝

    companion object {
        // 등수 판별
        fun determineRank(predicted: Set<Int>, actual: LottoTicket): Rank {

            val matchCount = predicted.intersect(actual.numbers).size

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