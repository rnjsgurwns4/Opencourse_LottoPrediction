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
import weka.core.Instances


// 훈련
fun trainModel(trainingHistory: List<LottoTicket>): Triple<Map<Int, Classifier>, Instances, FeatureEngineer> {
    val featureEngineer = FeatureEngineer()
    val trainingData = featureEngineer.createTrainingData(trainingHistory)

    val modelTrainer = LottoModelTrainer()
    modelTrainer.train(trainingData)

    val models = modelTrainer.getModels()
    val dataHeader = modelTrainer.dataHeader
    println("[Trainer] 훈련 완료.")
    return Triple(models, dataHeader, featureEngineer)
}

// 예측
fun runPrediction(
    predictor: LottoPredictor,
    history: List<LottoTicket>,
    latestDraws: List<LottoTicket>
): List<Int> {
    return predictor.predictNextDraw(history, latestDraws)
}


// Ktor 웹 서버 실행
lateinit var futurePredictionResult: List<Int>
lateinit var pastTestReportHtml: String
var latestDrwNoForFuture: Int = 0

fun main() {
    println("Kotlin ML 로또 서버 훈련 시작 (웹 서버 시작 전)")

    // 서버 시작 전에 모든 데이터를 미리 훈련/예측
    runBlocking {
        val dataManager = LottoDataManager()
        val fullHistory = dataManager.fetchAllHistory()

        if (fullHistory.size < 21) {
            println("오류: 데이터 부족. 서버를 시작할 수 없습니다.")
            return@runBlocking
        }

        // 미래(다음 주) 예측
        println("\n 미래 예측 모델 훈련 중 (1~${fullHistory.size}회차 사용)")
        latestDrwNoForFuture = fullHistory.last().drwNo

        val (modelsFuture, headerFuture, feFuture) = trainModel(fullHistory)
        val predictorFuture = LottoPredictor(modelsFuture, headerFuture, feFuture)
        futurePredictionResult = runPrediction(
            predictorFuture,
            fullHistory,
            fullHistory.takeLast(10)
        )
        println("미래 예측 번호 분석 완료: $futurePredictionResult")


        // 과거(가장 최근) 검증
        println("\n과거 검증 모델 훈련 중 (1~${fullHistory.size - 1}회차 사용)")
        val trainingHistoryPast = fullHistory.dropLast(1)
        val actualAnswer = fullHistory.last()
        val latest10DrawsPast = trainingHistoryPast.takeLast(10)

        val (modelsPast, headerPast, fePast) = trainModel(trainingHistoryPast)
        val predictorPast = LottoPredictor(modelsPast, headerPast, fePast)
        val pastPredictionResult = runPrediction(
            predictorPast,
            trainingHistoryPast,
            latest10DrawsPast
        )
        val pastRank = Rank.determineRank(pastPredictionResult.toSet(), actualAnswer)
        println("과거 검증 완료: $pastRank")

        pastTestReportHtml = generatePastReportHtml(pastPredictionResult, actualAnswer, pastRank)
    }

    println("\nKtor 웹 서버를 http://localhost:8080 에서 시작합니다.")

    // Ktor 웹 서버 실행 (8080 포트)
    embeddedServer(CIO, port = 8080) {
        routing {

            // 메인 페이지
            get("/") {
                call.respondHtml(HttpStatusCode.OK) {
                    head { title("ML 로또 예측기") }
                    body {
                        h1 { +"Kotlin ML 로또 예측기" }
                        p { +"미션: 낯선 도구(Kotlin+Weka)로 로또 예측 문제 해결하기" }

                        // 버튼 1: 미래 예측
                        form(action = "/predict", method = FormMethod.get) {
                            button(type = ButtonType.submit) {
                                +"[미래] 다음 회차 번호 예측하기"
                            }
                        }
                        br()

                        // 버튼 2: 과거 검증
                        form(action = "/test", method = FormMethod.get) {
                            button(type = ButtonType.submit) {
                                +"[과거] 가장 최근 회차 검증하기"
                            }
                        }
                    }
                }
            }

            // --- 페이지 1: 미래 예측 결과 ---
            get("/predict") {
                call.respondHtml(HttpStatusCode.OK) {
                    head { title("미래 예측 결과") }
                    body {
                        h1 { +"다음 회차 예측 번호" }
                        h2 { +"${futurePredictionResult}" }
                        p { +"(1~${latestDrwNoForFuture}회차 전체 데이터를 학습한 결과입니다.)" } // 1197은 예시
                        br()
                        a(href = "/") { +"뒤로가기" }
                    }
                }
            }

            // --- 페이지 2: 과거 검증 결과 ---
            get("/test") {
                call.respondText(pastTestReportHtml, ContentType.Text.Html)
            }
        }
    }.start(wait = true)
}

// 과거 검증용 HTML 생성
fun generatePastReportHtml(predicted: List<Int>, actual: LottoTicket, rank: Rank): String {

    return """
    <html>
        <head><title>과거 검증 결과</title></head>
        <style>
            body { font-family: sans-serif; }
            table { border-collapse: collapse; }
            th, td { border: 1px solid #ccc; padding: 8px; }
            th { background-color: #f4f4f4; }
        </style>
    <body>
        <h1>ML 모델 vs 가장 최근 회차(${actual.drwNo}회) 결과</h1>
        <table>
            <tr><th>모델 예측 번호</th><td>${predicted}</td></tr>
            <tr><th>실제 당첨 번호</th><td>${actual.numbers}</td></tr>
            <tr><th>실제 보너스</th><td>${actual.bonusNo}</td></tr>
        </table>
        <br>
        <h2>결과: $rank</h2>
        <br>
        <a href="/">뒤로가기</a>
    </body>
    </html>
    """.trimIndent()
}