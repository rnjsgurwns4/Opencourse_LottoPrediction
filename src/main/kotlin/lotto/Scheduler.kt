package lotto

import kotlinx.coroutines.*
import java.time.DayOfWeek
import java.time.LocalDateTime

//스케쥴러
fun startScheduler() {
    GlobalScope.launch {
        // 토요일 9시 즈음마다 업데이트
        while (isActive) {
            val now = LocalDateTime.now()
            if (now.dayOfWeek == DayOfWeek.SATURDAY && now.hour == 21 && now.minute in 15..25) {
                println("\n업데이트 시작")
                try {
                    TrainingService.performTotalTraining() // 서비스 호출
                    delay(1000L * 60 * 60 * 6)
                } catch (e: Exception) {
                    e.printStackTrace()
                    delay(1000L * 60 * 10)
                }
            }
            delay(1000L * 60)
        }
    }
}