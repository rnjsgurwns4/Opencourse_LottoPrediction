package lotto

import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.statement.*
import io.ktor.client.request.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

// 로또 데이터 가져오기
class LottoDataManager {

    private val jsonParser = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
    }

    // Ktor HTTP 클라이언트 설정
    private val client = HttpClient(CIO) {
        install(ContentNegotiation) {
            json(jsonParser)
        }
    }

    // API 응답을 파싱하기 위한 데이터 클래스
    @Serializable
    private data class ApiResponse(
        val returnValue: String,
        val drwNo: Int? = null,          // <-- = null 추가
        val drwNoDate: String? = null,   // <-- = null 추가
        val drwtNo1: Int? = null,        // <-- = null 추가
        val drwtNo2: Int? = null,        // <-- = null 추가
        val drwtNo3: Int? = null,        // <-- = null 추가
        val drwtNo4: Int? = null,        // <-- = null 추가
        val drwtNo5: Int? = null,        // <-- = null 추가
        val drwtNo6: Int? = null,        // <-- = null 추가
        val bnusNo: Int? = null
    )

    // 로또 내역
    suspend fun fetchAllHistory(): List<LottoTicket> {
        val history = mutableListOf<LottoTicket>()
        var currentDrwNo = 1

        println("1회차부터 과거 데이터 수집을 시작합니다")

        while (true) {
            try {
                val url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo=$currentDrwNo"

                val responseString = client.get(url).bodyAsText()
                val response = jsonParser.decodeFromString<ApiResponse>(responseString)

                if (response.returnValue == "fail" || response.drwNo == null) {
                    println("총 ${currentDrwNo - 1}회차까지의 데이터를 수집했습니다.")
                    break
                }

                val numbers = setOfNotNull(
                    response.drwtNo1, response.drwtNo2, response.drwtNo3,
                    response.drwtNo4, response.drwtNo5, response.drwtNo6
                )

                if (numbers.size == 6 && response.bnusNo != null) {
                    history.add(LottoTicket(response.drwNo, numbers, response.bnusNo))
                }

                if (currentDrwNo % 100 == 0) {
                    println("... $currentDrwNo 회차 수집 중 ...")
                }
                currentDrwNo++

            } catch (e: Exception) {
                println("데이터 수집 중 오류 발생 (회차: $currentDrwNo): ${e.message}")
                break // 오류 발생 시 중단
            }
        }
        client.close()
        return history
    }
}