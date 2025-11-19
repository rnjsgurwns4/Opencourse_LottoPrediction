package lotto


 // 애플리케이션의 전역 상태(훈련된 모델, 데이터, 리포트 등)를 관리하는 객체
object AppState {
    lateinit var bestPredictor_RankStrategy: LottoPredictor
    lateinit var bestPredictor_WinsStrategy: LottoPredictor

    lateinit var championRank: ModelScore
    lateinit var championWins: ModelScore

    lateinit var fullHistoryForPredict: List<LottoTicket>
    lateinit var latestDrawsShortForPredict: List<LottoTicket>
    lateinit var latestDrawsMidForPredict: List<LottoTicket>

    var lastDrawNo: Int = 0
    lateinit var pastTestReportHtml: String

    var isInitialized: Boolean = false
}

data class ModelScore(
    val modelName: String,
    val bestRank: Rank,
    val totalWins: Int
)

enum class SelectionStrategy {
    BEST_RANK_FIRST,
    MOST_WINS_FIRST
}