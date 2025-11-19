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

import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString

fun main() {
    // 초기 훈련
    runBlocking { TrainingService.performTotalTraining() }

    // 스케줄러 시작
    startScheduler()

    // 웹 서버
    println("\nKtor 웹 서버를 http://localhost:8080 에서 시작합니다.")
    embeddedServer(CIO, port = 8080) {
        routing {

            // 메인 페이지
            get("/") {
                if (!AppState.isInitialized) {
                    call.respondText("서버 준비 중입니다. 잠시 후 새로고침 해주세요.")
                    return@get
                }
                call.respondHtml(HttpStatusCode.OK) {
                    head { title("ML 로또 예측기"); style { +globalStyles } }
                    body {
                        h1 { +"Kotlin ML 로또 예측기" }
                        p { +"미션: 낯선 도구(Kotlin+Weka)로 로또 예측 문제 해결하기" }

                        h2 { +"[미래] 다음 회차 번호 예측하기" }
                        form(action = "/predict_strategy", method = FormMethod.get) {
                            button(type = ButtonType.submit) { +"예측 전략 선택하기" }
                        }

                        h2 { +"[과거] 가장 최근 회차 검증하기" }
                        form(action = "/test", method = FormMethod.get) {
                            button(type = ButtonType.submit) { +"결과 보기" }
                        }

                        h2 { +"[통계] 데이터 시각화" }
                        form(action = "/stats", method = FormMethod.get) {
                            button(type = ButtonType.submit) { +"학습 데이터 그래프 보기" }
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
                            button(type = ButtonType.submit) { +"① 최고 등수 우선" }
                        }
                        br()

                        h2 { +"전략 ②: 총 당첨 횟수 우선" }

                        form(action = "/predict_run", method = FormMethod.get) {
                            input(type = InputType.hidden, name = "strategy") { value = "MOST_WINS_FIRST" }
                            button(type = ButtonType.submit) { +"② 총 당첨 횟수 우선" }
                        }

                        br()
                        hr()
                        a(href = "/") { +"메인으로 돌아가기" }
                    }
                }
            }

            // 개수 입력
            get("/predict_run") {
                val strategyStr = call.request.queryParameters["strategy"] ?: "BEST_RANK_FIRST"

                val (championName, championScore) = if (strategyStr == "MOST_WINS_FIRST") {
                    AppState.championWins.modelName to AppState.championWins
                } else {
                    AppState.championRank.modelName to AppState.championRank
                }

                call.respondHtml(HttpStatusCode.OK) {
                    head { title("설정"); style { +globalStyles } }
                    body {
                        h1 { +"예측 설정" }
                        p { +"사용된 모델: $championName (성적: ${championScore.bestRank})" }
                        form(action = "/predict_results", method = FormMethod.get) {
                            input(type = InputType.hidden, name = "strategy") { value = strategyStr }
                            label { +"세트 개수: " }
                            input(type = InputType.number, name = "n") { value = "5"; min = "1"; max = "10" }
                            button(type = ButtonType.submit) { +"예측 실행" }
                        }
                        br()
                        a(href = "/predict_strategy") { +"전략 다시 선택하기" }
                    }
                }
            }

            // 결과 확인
            get("/predict_results") {
                val n = call.request.queryParameters["n"]?.toIntOrNull() ?: 5
                val strategyStr = call.request.queryParameters["strategy"]

                val (predictor, name) = if (strategyStr == "MOST_WINS_FIRST") {
                    AppState.bestPredictor_WinsStrategy to AppState.championWins.modelName
                } else {
                    AppState.bestPredictor_RankStrategy to AppState.championRank.modelName
                }

                val results = predictor.predictNextDraw(
                    AppState.fullHistoryForPredict,
                    AppState.latestDrawsShortForPredict,
                    AppState.latestDrawsMidForPredict,
                    n
                )

                call.respondHtml {
                    head { title("결과"); style { +globalStyles } }
                    body {
                        h1 { +"다음 회차(${AppState.lastDrawNo + 1}회) 예측 결과" }
                        p { +"모델: $name" }
                        ul {
                            results.forEach { set ->
                                li {
                                    set.forEach { num -> span(classes = "lotto-ball ball-${(num-1)/10 + 1}") { +"$num" } }
                                }
                            }
                        }
                        a(href = "/") { +"메인으로 돌아가기" }
                    }
                }
            }

            // 과거 검증
            get("/test") {
                call.respondText(AppState.pastTestReportHtml, ContentType.Text.Html)
            }

            // 통계 (Chart.js)
            get("/stats") {

                val featureEngineer = AppState.bestPredictor_RankStrategy.featureEngineer

                val featureMap = featureEngineer.createCurrentFeaturesForPrediction(
                    AppState.fullHistoryForPredict,
                    AppState.latestDrawsMidForPredict,
                    AppState.latestDrawsMidForPredict
                )

                // Chart.js에 주입할 5개의 데이터 리스트 생성
                val labels = (1..45).toList() // X축 (1~45번)
                val dataRecency = (1..45).map { featureMap[it]?.get("recency") as Int }
                val dataFreqShort = (1..45).map { featureMap[it]?.get("freq_short") as Int }
                val dataFreqMid = (1..45).map { featureMap[it]?.get("freq_mid") as Int }
                val dataFreqTotalMain = (1..45).map { featureMap[it]?.get("freq_total_main") as Int }
                val dataFreqTotalBonus = (1..45).map { featureMap[it]?.get("freq_total_bonus") as Int }

                call.respondHtml(HttpStatusCode.OK) {
                    head {
                        title("학습 데이터 시각화")
                        style { +globalStyles }
                        // Chart.js CDN 추가
                        script(src = "https://cdn.jsdelivr.net/npm/chart.js") {}
                    }
                    body {
                        h1 { +"현재 학습 데이터 (특성) 시각화" }
                        p { +"ML 모델은 이 5가지 특성 그래프의 패턴을 학습하여 다음 회차를 예측합니다." }

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

                        // Kotlin 데이터를 JS 변수로 주입하고 차트 그리기
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
        }
    }.start(wait = true)
}