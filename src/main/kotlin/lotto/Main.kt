package lotto

import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.cio.*
import io.ktor.server.html.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.coroutines.runBlocking
import kotlinx.html.*
import weka.classifiers.Classifier
import weka.classifiers.functions.Logistic
import weka.classifiers.trees.J48
import weka.classifiers.trees.RandomForest
import weka.core.Instances
import org.jetbrains.kotlinx.dataframe.DataFrame
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json


// 훈련
fun trainAllModels(
    trainingData: DataFrame<*>,
    modelsToTrain: Map<String, Classifier>
): Pair<Map<String, Map<Int, Classifier>>, Instances> {

    val trainer = LottoModelTrainer()
    val allTrainedModels = mutableMapOf<String, Map<Int, Classifier>>()

    // 첫 번째 모델을 훈련시켜서 dataHeader 추출
    val firstModelName = modelsToTrain.keys.first()
    trainer.train(trainingData, modelsToTrain[firstModelName]!!)
    allTrainedModels[firstModelName] = trainer.getModels()
    val dataHeader = trainer.dataHeader

    // 나머지 모델들을 훈련
    modelsToTrain.keys.drop(1).forEach { modelName ->
        trainer.train(trainingData, modelsToTrain[modelName]!!)
        allTrainedModels[modelName] = trainer.getModels()
    }

    return Pair(allTrainedModels, dataHeader)
}

// 2가지 선택 전략 정의
enum class SelectionStrategy {
    BEST_RANK_FIRST, // 1순위: 최고 등수, 2순위: 총 당첨 횟수
    MOST_WINS_FIRST  // 1순위: 총 당첨 횟수, 2순위: 최고 등수
}

// 각 모델의 성적
data class ModelScore(
    val modelName: String,
    val bestRank: Rank,  // 3번의 시도 중 최고 등수
    val totalWins: Int   // 3번의 시도 중 꽝이 아닌 총 횟수
)

// 성적이 가장 좋은 모델 찾기
fun findBestModel(
    ranksMap: Map<String, List<Rank>>,
    strategy: SelectionStrategy
): ModelScore {

    // 1. 각 모델의 성적 리스트 생성
    val scores = ranksMap.map { (name, rankList) ->
        val wins = rankList.filter { it != Rank.NONE } // 꽝이 아닌 것만 필터링
        ModelScore(
            modelName = name,
            bestRank = wins.minByOrNull { it.ordinal } ?: Rank.NONE, // 최고 등수
            totalWins = wins.size // 총 당첨 횟수
        )
    }

    // 선택된 전략에 따라 성적 리스트 정렬
    val sortedScores = scores.sortedWith(
        when (strategy) {
            SelectionStrategy.BEST_RANK_FIRST ->
                compareBy<ModelScore> { it.bestRank.ordinal }
                    .thenByDescending { it.totalWins }

            SelectionStrategy.MOST_WINS_FIRST ->
                compareByDescending<ModelScore> { it.totalWins }
                    .thenBy { it.bestRank.ordinal }
        }
    )

    return sortedScores.first()
}


// Ktor 웹 서버 실행
lateinit var bestPredictor_RankStrategy: LottoPredictor
lateinit var championRank: ModelScore
lateinit var bestPredictor_WinsStrategy: LottoPredictor
lateinit var championWins: ModelScore

lateinit var fullHistoryForPredict: List<LottoTicket>
lateinit var latestDrawsShortForPredict: List<LottoTicket>
lateinit var latestDrawsMidForPredict: List<LottoTicket>
var lastDrawNo: Int = 0
lateinit var pastTestReportHtml: String

fun main() {
    println("Kotlin ML 로또 서버 훈련 시작 (웹 서버 시작 전)")

    val modelsToTrain = mapOf(
        "Logistic" to Logistic(),
        "RandomForest" to RandomForest(),
        "J48" to J48()
    )
    // 서버 시작 전에 모든 데이터를 미리 훈련/예측
    runBlocking {
        val dataManager = LottoDataManager()
        val fullHistory = dataManager.fetchAllHistory()
        lastDrawNo = fullHistory.last().drwNo

        if (fullHistory.size < 26) {
            println("오류: 데이터 부족. 서버를 시작할 수 없습니다.")
            return@runBlocking
        }

        // 미래(다음 주) 예측
        println("\n과거 검증 시작")
        val trainingHistoryPast = fullHistory.dropLast(1)
        val actualAnswer = fullHistory.last()
        val latestDrawsShortPast = trainingHistoryPast.takeLast(10)
        val latestDrawsMidPast = trainingHistoryPast.takeLast(25)

        val fePast = FeatureEngineer()
        val tdPast = fePast.createTrainingData(trainingHistoryPast)

        // 모든 모델 훈련 (Logistic, RandomForest, J48)
        val (pastTrainedModels, pastHeader) = trainAllModels(tdPast, modelsToTrain)

        // 모든 모델로 예측기 생성 및 예측 실행
        val pastPredictors = pastTrainedModels.mapValues { (_, modelSet) ->
            LottoPredictor(modelSet, pastHeader, fePast)
        }
        val pastResults = pastPredictors.mapValues { (name, predictor) ->
            println("[Main] '${name}' 모델로 과거 예측 중")
            predictor.predictNextDraw(
                trainingHistoryPast,
                latestDrawsShortPast,
                latestDrawsMidPast,
                3
            )
        }

        // 모든 모델의 등수 계산
        val pastRanks = pastResults.mapValues { (name, sets) ->
            sets.map { numbers -> Rank.determineRank(numbers.toSet(), actualAnswer) }
        }

        championRank = findBestModel(pastRanks, SelectionStrategy.BEST_RANK_FIRST)
        championWins = findBestModel(pastRanks, SelectionStrategy.MOST_WINS_FIRST)

        println("과거 검증 완료!")
        println("[최고 등수]: ${championRank.modelName} (성적: ${championRank.bestRank} / ${championRank.totalWins}회)")
        println("[최다 당첨]: ${championWins.modelName} (성적: ${championWins.bestRank} / ${championWins.totalWins}회)")

        // /test 페이지에 보여줄 HTML 리포트 미리 생성
        pastTestReportHtml = generatePastReportHtml(pastResults, pastRanks, actualAnswer)


        // 미래 예측 (1등으로 뽑힌 모델만 사용) ---
        println("\n미래 예측기 2개 훈련/캐시 시작")

        val feFuture = FeatureEngineer()
        // 전체 데이터로 훈련 데이터 다시 생성
        val tdFuture = feFuture.createTrainingData(fullHistory)

        // 1등 모델 하나만 다시 훈련
        val championsToTrainNames = setOf(championRank.modelName, championWins.modelName)

        // 전체 데이터로 재훈련
        val futureBaseModels = modelsToTrain.filterKeys { it in championsToTrainNames }
        val (futureTrainedModels, futureHeader) = trainAllModels(tdFuture, futureBaseModels)

        bestPredictor_RankStrategy = LottoPredictor(
            futureTrainedModels[championRank.modelName]!!,
            futureHeader, feFuture
        )
        bestPredictor_WinsStrategy = LottoPredictor(
            futureTrainedModels[championWins.modelName]!!,
            futureHeader, feFuture
        )

        fullHistoryForPredict = fullHistory
        latestDrawsShortForPredict = fullHistory.takeLast(10)
        latestDrawsMidForPredict = fullHistory.takeLast(25)

        println("미래 예측 완료")
    }

    println("\nKtor 웹 서버를 http://localhost:8080 에서 시작합니다.")

    // Ktor 웹 서버 실행 (8080 포트)
    embeddedServer(CIO, port = 8080) {
        routing {
            // 메인 페이지
            get("/") {
                call.respondHtml(HttpStatusCode.OK) {
                    head {
                        title("ML 로또 예측기")
                        style { +globalStyles }
                    }
                    body {
                        h1 { +"Kotlin ML 로또 예측기" }
                        p { +"미션: 낯선 도구(Kotlin+Weka)로 로또 예측 문제 해결하기" }

                        h2 { +"[미래] 다음 회차 번호 예측하기" }
                        p { +"미래 번호를 예측합니다." }
                        form(action = "/predict_strategy", method = FormMethod.get) {
                            button(type = ButtonType.submit) {
                                +"예측 전략 선택하기"
                            }
                        }

                        h2 { +"[과거] 가장 최근 회차 검증하기" }
                        form(action = "/test", method = FormMethod.get) {
                            button(type = ButtonType.submit) {
                                +"결과 보기"
                            }
                        }

                        h2 { +"[통계] 현재 학습 데이터(특성) 보기" }
                        p { +"모델이 다음 회차를 예측하기 위해 사용하는 특성 값을 시각화합니다." }
                        form(action = "/stats", method = FormMethod.get) {
                            button(type = ButtonType.submit) {
                                +"학습 데이터 그래프 보기"
                            }
                        }
                    }
                }
            }

            get("/stats") {

                // 1. 캐시된 1등 예측기('최고 등수' 기준)의 '특성 엔지니어'를 사용
                //    (어떤 챔피언이든 featureEngineer는 동일하게 작동함)
                val featureEngineer = bestPredictor_RankStrategy.featureEngineer

                // 2. '현재 시점'의 1~45번 특성 맵을 가져옴
                val featureMap = featureEngineer.createCurrentFeaturesForPrediction(
                    fullHistoryForPredict,
                    latestDrawsShortForPredict,
                    latestDrawsMidForPredict
                )

                // 3. Chart.js에 주입할 5개의 데이터 리스트 생성
                val labels = (1..45).toList() // X축 (1~45번)
                val dataRecency = (1..45).map { featureMap[it]?.get("recency") as Int }
                val dataFreqShort = (1..45).map { featureMap[it]?.get("freq_short") as Int }
                val dataFreqMid = (1..45).map { featureMap[it]?.get("freq_mid") as Int }
                val dataFreqTotalMain = (1..45).map { featureMap[it]?.get("freq_total_main") as Int }
                val dataFreqTotalBonus = (1..45).map { featureMap[it]?.get("freq_total_bonus") as Int }

                // 4. HTML 응답 (Chart.js 포함)
                call.respondHtml(HttpStatusCode.OK) {
                    head {
                        title("학습 데이터 시각화")
                        style { +globalStyles } // 공통 스타일
                        // Chart.js CDN 추가
                        script(src = "https://cdn.jsdelivr.net/npm/chart.js") {}
                    }
                    body {
                        h1 { +"현재 학습 데이터 (특성) 시각화" }
                        p { +"ML 모델은 이 5가지 특성 그래프의 패턴을 학습하여 다음 회차를 예측합니다." }

                        // 차트를 그릴 5개의 <canvas> 태그
                        h2 { +"1. Recency (미출현 기간)" }
                        p { +"(0: 지난주에 나옴, 25: 최근 25주간 안 나옴)" }
                        canvas { id = "chartRecency" }

                        h2 { +"2. Freq. Short (단기 빈도)" }
                        p { +"최근 10회간 메인 번호로 나온 횟수" }
                        canvas { id = "chartFreqShort" }

                        h2 { +"3. Freq. Mid (중기 빈도)" }
                        p { +"최근 25회간 메인 번호로 나온 횟수" }
                        canvas { id = "chartFreqMid" }

                        h2 { +"4. Freq. Total Main (누적 메인 빈도)" }
                        p { +"1회차부터 현재까지 메인 번호로 나온 총 횟수" }
                        canvas { id = "chartFreqTotalMain" }

                        h2 { +"5. Freq. Total Bonus (누적 보너스 빈도)" }
                        p { +"1회차부터 현재까지 보너스 번호로 나온 총 횟수" }
                        canvas { id = "chartFreqTotalBonus" }

                        br()
                        a(href = "/") { +"메인으로 돌아가기" }

                        // ★ 5. (신규) Kotlin 데이터를 JS 변수로 주입하고 차트 그리기
                        script {
                            unsafe {
                                // Kotlin List를 JavaScript 배열 문자열로 변환
                                raw("""
                                const labels = ${Json.encodeToString(labels)};
                                const dataRecency = ${Json.encodeToString(dataRecency)};
                                const dataFreqShort = ${Json.encodeToString(dataFreqShort)};
                                const dataFreqMid = ${Json.encodeToString(dataFreqMid)};
                                const dataFreqTotalMain = ${Json.encodeToString(dataFreqTotalMain)};
                                const dataFreqTotalBonus = ${Json.encodeToString(dataFreqTotalBonus)};
                                
                                // 차트 생성 헬퍼 함수
                                function createChart(canvasId, chartLabel, data) {
                                    new Chart(document.getElementById(canvasId), {
                                        type: 'bar',
                                        data: {
                                            labels: labels,
                                            datasets: [{
                                                label: chartLabel,
                                                data: data,
                                                backgroundColor: 'rgba(0, 123, 255, 0.7)',
                                            }]
                                        },
                                        options: {
                                            scales: {
                                                x: { title: { display: true, text: '로또 번호' } },
                                                y: { beginAtZero: true, title: { display: true, text: '값' } }
                                            }
                                        }
                                    });
                                }
                                
                                // 5개 차트 그리기
                                createChart('chartRecency', 'Recency (미출현 기간)', dataRecency);
                                createChart('chartFreqShort', '최근 10회 빈도', dataFreqShort);
                                createChart('chartFreqMid', '최근 25회 빈도', dataFreqMid);
                                createChart('chartFreqTotalMain', '누적 메인 빈도', dataFreqTotalMain);
                                createChart('chartFreqTotalBonus', '누적 보너스 빈도', dataFreqTotalBonus);
                                """.trimIndent())
                            }
                        }
                    }
                }
            }

            // 전략 선택
            get("/predict_strategy") {
                call.respondHtml(HttpStatusCode.OK) {
                    head { title("전략 선택"); style { +globalStyles } }
                    body {
                        h1 { +"[미래] 예측 전략 선택" }
                        p { +"사용할 전략을 선택하세요:" }

                        h2 { +"전략 ①: 최고 등수 우선" }

                        form(action = "/predict_run", method = FormMethod.get) {
                            input(type = InputType.hidden, name = "strategy") { value = "BEST_RANK_FIRST" }
                            button(type = ButtonType.submit) { +"①번 전략 선택" }
                        }
                        br()

                        h2 { +"전략 ②: 총 당첨 횟수 우선" }

                        form(action = "/predict_run", method = FormMethod.get) {
                            input(type = InputType.hidden, name = "strategy") { value = "MOST_WINS_FIRST" }
                            button(type = ButtonType.submit) { +"②번 전략 선택" }
                        }
                        br()
                        hr()
                        a(href = "/") { +"메인으로 돌아가기" }
                    }
                }
            }

            // 미래 예측 결과
            get("/predict_run") {
                val strategy = call.request.queryParameters["strategy"] ?: "BEST_RANK_FIRST"
                val (championName, championScore) = if (strategy == "MOST_WINS_FIRST") {
                    championWins.modelName to championWins
                } else {
                    championRank.modelName to championRank
                }

                call.respondHtml(HttpStatusCode.OK) {
                    head { title("세트 개수 입력"); style { +globalStyles } }
                    body {
                        h1 { +"[${if (strategy == "MOST_WINS_FIRST") "총 당첨 횟수" else "최고 등수"} 우선] 전략" }
                        p { +"사용한 모델: $championName (성적: ${championScore.bestRank} / ${championScore.totalWins}회)" }
                        hr()

                        form(action = "/predict_results", method = FormMethod.get) {
                            input(type = InputType.hidden, name = "strategy") { value = strategy }

                            label { +"생성할 로또 세트 개수 (1~10): " }
                            input(type = InputType.number, name = "n") {
                                value = "5"
                                min = "1"
                                max = "10"
                            }
                            button(type = ButtonType.submit) { +"최종 예측 실행" }
                        }
                        br()
                        a(href = "/predict_strategy") { +"전략 다시 선택하기" }
                    }
                }
            }

            get("/predict_results") {
                val n = call.request.queryParameters["n"]?.toIntOrNull()?.coerceIn(1, 10) ?: 1
                val strategy = call.request.queryParameters["strategy"]

                val (predictorToUse, championName, strategyName) =
                    if (strategy == "MOST_WINS_FIRST") {
                        Triple(bestPredictor_WinsStrategy, championWins.modelName, "총 당첨 횟수 우선")
                    } else {
                        Triple(bestPredictor_RankStrategy, championRank.modelName, "최고 등수 우선")
                    }

                // 예측 실행
                val resultSets = predictorToUse.predictNextDraw(
                    fullHistoryForPredict,
                    latestDrawsShortForPredict,
                    latestDrawsMidForPredict,
                    n
                )

                // HTML 응답
                call.respondHtml(HttpStatusCode.OK) {
                    head { title("미래 예측 결과"); style { +globalStyles } }
                    body {
                        h1 { +"다음 회차(${lastDrawNo + 1}회) 예측 번호 ($n 세트)" }
                        p { +"사용한 전략: $strategyName" }
                        p { +"사용한 모델: $championName" }

                        ul { resultSets.forEach { set -> li { b { +"${set}" } } } }
                        br()
                        a(href = "/predict_run?strategy=$strategy") { +"세트 개수 다시 입력" }
                        br()
                        a(href = "/") { +"메인으로 돌아가기" }
                    }
                }
            }

            // 과거 검증 결과
            get("/test") {
                call.respondText(pastTestReportHtml, ContentType.Text.Html)
            }
        }
    }.start(wait = true)
}

fun generatePastReportHtml(
    predictedMap: Map<String, List<List<Int>>>, // 세트가 담긴 List
    ranksMap: Map<String, List<Rank>>,       // 세트의 등급 List
    actual: LottoTicket,
): String {

    val tableRows = predictedMap.keys.joinToString("") { modelName ->
        val sets = predictedMap[modelName]!!
        val ranks = ranksMap[modelName]!!

        // 모델별 성적표 계산
        val wins = ranks.filter { it != Rank.NONE }
        val bestRank = wins.minByOrNull { it.ordinal } ?: Rank.NONE
        val totalWins = wins.size

        // HTML 생성
        val attemptsHtml = (0..2).joinToString("") { i ->
            "<tr><td>${sets[i]}</td><td><strong>${ranks[i]}</strong></td></tr>"
        }

        // 최종 HTML Row 생성
        """
        <tr>
            <td rowspan="4"><strong>${modelName}</strong></td>
            ${attemptsHtml.substring(4)} </tr>
        <tr style="background-color: #f8f8f8;">
            <td><strong>모델 성적 (Best / Total)</strong></td>
            <td><strong>${bestRank} / ${totalWins} 회</strong></td>
        </tr>
        """
    }

    return """
    <html>
        <head>
            <title>과거 검증 결과</title>
            <style>${globalStyles}</style>
        </head>

    <body>
        <h1>ML 모델 간 비교 (vs ${actual.drwNo}회차)</h1>
        
        <h2>[검증 대상]</h2>
        <table>
            <tr><th>실제 당첨 번호</th><td>${actual.numbers}</td></tr>
            <tr><th>실제 보너스</th><td>${actual.bonusNo}</td></tr>
        </table>
        <br>
        
        <h2>[모델별 3세트 예측 및 성적]</h2>
        <table>
            <tr>
                <th>모델 이름</th>
                <th>예측 번호 (3회 시도)</th>
                <th>결과 (등수)</th>
            </tr>
            $tableRows
        </table>
        <br>
        <h3>🏆 가장 효과적인 모델 (미래 예측에 사용)</h3>
        <ul>
        </ul>
        <br>
        <a href="/">뒤로가기</a>
    </body>
    </html>
    """.trimIndent()
}

// css
val globalStyles = """
    body { 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
        max-width: 900px; 
        margin: 20px auto; 
        background-color: #f9f9f9;
        color: #333;
    }
    h1, h2 { color: #0056b3; }
    h2 { border-top: 2px solid #eee; padding-top: 15px; }
    table { 
        border-collapse: collapse; 
        width: 100%; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background: #fff;
    }
    th, td { 
        border: 1px solid #ddd; 
        padding: 12px; 
        text-align: left; 
    }
    th { background-color: #f4f4f4; }
    td[rowspan] { 
        background-color: #fdfdfd; 
        font-weight: bold; 
        vertical-align: top; 
        text-align: center;
    }
    button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
    }
    button:hover { background-color: #0056b3; }
    input[type="number"] { padding: 8px; border-radius: 4px; border: 1px solid #ccc; }
    ul { list-style: none; padding-left: 0; }
    li { 
        background: #fff; 
        border: 1px solid #eee; 
        padding: 10px; 
        margin-bottom: 5px; 
        border-radius: 4px;
        font-family: 'Courier New', Courier, monospace;
    }
    a { color: #007bff; text-decoration: none; }
    a:hover { text-decoration: underline; }
"""