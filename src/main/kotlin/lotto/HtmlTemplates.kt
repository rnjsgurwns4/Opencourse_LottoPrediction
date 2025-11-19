package lotto

// CSS 스타일
val globalStyles = """
    body { 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
        max-width: 900px; margin: 20px auto; background-color: #f9f9f9; color: #333;
    }
    h1, h2 { color: #0056b3; }
    h2 { border-top: 2px solid #eee; padding-top: 15px; }
    table { border-collapse: collapse; width: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background: #fff; }
    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
    th { background-color: #f4f4f4; }
    td[rowspan] { background-color: #fdfdfd; font-weight: bold; vertical-align: top; text-align: center; }
    button { background-color: #007bff; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; font-weight: bold; }
    button:hover { background-color: #0056b3; }
    input { padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
    ul { list-style: none; padding-left: 0; }
    li { background: #fff; border: 1px solid #eee; padding: 10px; margin-bottom: 5px; border-radius: 4px; font-family: monospace; }
    a { color: #007bff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    
    .lotto-ball { display: inline-block; width: 30px; height: 30px; line-height: 30px; border-radius: 50%; color: white; text-align: center; font-weight: bold; margin-right: 5px; text-shadow: 1px 1px 1px rgba(0,0,0,0.3); }
    .ball-1 { background-color: #fbc400; } .ball-2 { background-color: #69c8f2; } .ball-3 { background-color: #ff7272; } .ball-4 { background-color: #aaa; } .ball-5 { background-color: #b0d840; }
"""

// 과거 검증 리포트 HTML 생성 함수
fun generatePastReportHtml(
    predictedMap: Map<String, List<List<Int>>>,
    ranksMap: Map<String, List<Rank>>,
    actual: LottoTicket,
    championRank: ModelScore,
    championWins: ModelScore
): String {
    val tableRows = predictedMap.keys.joinToString("") { modelName ->
        val sets = predictedMap[modelName]!!
        val ranks = ranksMap[modelName]!!
        val attemptsHtml = (0..2).joinToString("") { i ->
            "<tr><td>${sets[i]}</td><td><strong>${ranks[i]}</strong></td></tr>"
        }
        """
        <tr><td rowspan="4"><strong>${modelName}</strong></td>${attemptsHtml.substring(4)}</tr>
        <tr style="background-color: #f8f8f8;"><td><strong>성적</strong></td><td><strong>Best: ${ranks.minByOrNull { it.ordinal }}</strong></td></tr>
        """
    }
    return """
    <html><head><title>과거 검증 결과</title><style>$globalStyles</style></head>
    <body>
        <h1>ML 모델 검증 (vs ${actual.drwNo}회차)</h1>
        <h2>[검증 대상]</h2>
        <table><tr><th>당첨 번호</th><td>${actual.numbers} + ${actual.bonusNo}</td></tr></table>
        <br>
        <h2>[모델별 성적]</h2>
        <table><tr><th>모델</th><th>예측</th><th>결과</th></tr>$tableRows</table>
        <br>
        <h3> 선택된 모델: ${championRank.modelName} (Rank), ${championWins.modelName} (Wins)</h3>
        <a href="/">뒤로가기</a>
    </body></html>
    """.trimIndent()
}