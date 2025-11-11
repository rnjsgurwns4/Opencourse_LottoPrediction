// src/main/kotlin/lotto/LottoDataManager.kt
package lotto

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.statement.*
import io.ktor.client.request.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * 동행복권 API에서 과거 로또 당첨 데이터를 가져오는 클래스
 */
class LottoDataManager {

    private val jsonParser = Json {
        ignoreUnknownKeys = true // API 응답의 모든 필드를 사용하지 않음
        coerceInputValues = true
    }

    // Ktor HTTP 클라이언트 설정 (JSON 파싱 포함)
    private val client = HttpClient(CIO) {
        install(ContentNegotiation) {
            json(jsonParser)
        }
    }

    // API 응답을 파싱하기 위한 데이터 클래스 (내부용)
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

    /**
     * 1회차부터 현재까지의 모든 당첨 내역을 가져옵니다.
     * @return LottoTicket 리스트
     */
    suspend fun fetchAllHistory(): List<LottoTicket> {
        val history = mutableListOf<LottoTicket>()
        var currentDrwNo = 1

        println("1회차부터 과거 데이터 수집을 시작합니다...")

        while (true) {
            try {
                val url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo=$currentDrwNo"


                val responseString = client.get(url).bodyAsText()

                // ★ 4. 받아온 텍스트를 수동으로 파싱
                val response = jsonParser.decodeFromString<ApiResponse>(responseString)

                // "fail" 응답이 오면, 최신 회차까지 수집 완료한 것
                if (response.returnValue == "fail" || response.drwNo == null) {
                    println("총 ${currentDrwNo - 1}회차까지의 데이터를 수집했습니다.")
                    break
                }

                // API 응답에서 6개 번호를 Set으로 변환
                val numbers = setOfNotNull(
                    response.drwtNo1, response.drwtNo2, response.drwtNo3,
                    response.drwtNo4, response.drwtNo5, response.drwtNo6
                )

                if (numbers.size == 6 && response.bnusNo != null) {
                    history.add(LottoTicket(response.drwNo, numbers))
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
        client.close() // 클라이언트 종료
        return history
    }

}
