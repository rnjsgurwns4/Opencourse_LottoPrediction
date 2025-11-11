// src/main/kotlin/lotto/Main.kt
package lotto

import kotlinx.coroutines.runBlocking

fun main() = runBlocking { // Ktor(suspend 함수)를 실행하기 위해 runBlocking 사용

    // 1. 데이터 수집
    val dataManager = LottoDataManager()
    val fullHistory = dataManager.fetchAllHistory()

    if (fullHistory.size < 21) { // 최소 데이터 20개 (훈련 10 + 특성 10)
        println("오류: 학습에 필요한 최소 데이터(20회차)가 부족합니다.")
        return@runBlocking
    }

    // 2. 데이터 분리 (미래 데이터 유출 방지)
    // 훈련: 1회차 ~ (마지막-10) 회차
    val trainingHistory = fullHistory.dropLast(10)
    // 특성 계산용 최신 데이터: 마지막 10회차
    val latest10Draws = fullHistory.takeLast(10)

    // 3. 특성 공학
    val featureEngineer = FeatureEngineer()
    val trainingData = featureEngineer.createTrainingData(trainingHistory)


    // 4. 모델 훈련
    val modelTrainer = LottoModelTrainer()
    modelTrainer.train(trainingData) // 45개 모델 훈련

    // 5. 훈련된 결과 가져오기
    val models = modelTrainer.getModels()
    val dataHeader = modelTrainer.dataHeader // 훈련에 사용된 데이터 구조(헤더)

    // 6. 예측기 생성 및 예측 실행
    val predictor = LottoPredictor(models, dataHeader, featureEngineer)
    val predictedNumbers = predictor.predictTop6(
        fullHistory, // 누적 통계용 (1~최신)
        latest10Draws // 최근 특성용 (최신-10~최신)
    )

    println("\n---  최종 예측 결과 ---")
    println("Kotlin ML 시스템이 예측한 다음 회차 번호는")
    println(">>> $predictedNumbers <<<")
    println("\n(경고: 이 예측은 통계적 패턴 학습에 기반하며, 실제 당첨을 보장하지 않습니다.)")
}