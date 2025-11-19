package lotto

import org.jetbrains.kotlinx.dataframe.DataFrame
import weka.classifiers.Classifier
import weka.classifiers.functions.Logistic
import weka.classifiers.trees.J48
import weka.classifiers.trees.RandomForest
import weka.core.Instances

object TrainingService {

    // 훈련할 모델 목록 설정
    private val modelsToTrain: Map<String, Classifier> = mapOf(
        "Logistic" to Logistic(),
        "RandomForest_100" to RandomForest(),
        "RandomForest_500" to RandomForest().apply { numIterations = 500 },
        "J48_Pruned" to J48(),
        "J48_Unpruned" to J48().apply { unpruned = true }
    )


    // 데이터 수집, 모델 선발, 재훈련, 상태 업데이트
    suspend fun performTotalTraining() {
        val dataManager = LottoDataManager()
        val fullHistory = dataManager.fetchAllHistory()

        if (fullHistory.size < 26) {
            println("데이터 부족으로 훈련 중단.")
            return
        }

        // 상태 업데이트 (최신 회차)
        AppState.lastDrawNo = fullHistory.last().drwNo

        // 과거 검증
        println("\n과거 검증 및 모델 선발")
        val trainingHistoryPast = fullHistory.dropLast(1)
        val actualAnswer = fullHistory.last()

        val fePast = FeatureEngineer()
        val tdPast = fePast.createTrainingData(trainingHistoryPast)
        val (pastTrainedModels, pastHeader) = trainAllModels(tdPast, modelsToTrain)

        val pastPredictors = pastTrainedModels.mapValues { (_, modelSet) ->
            LottoPredictor(modelSet, pastHeader, fePast)
        }
        // 3세트 예측 실행
        val pastResults = pastPredictors.mapValues { (_, predictor) ->
            predictor.predictNextDraw(
                trainingHistoryPast,
                trainingHistoryPast.takeLast(10),
                trainingHistoryPast.takeLast(25),
                3
            )
        }
        val pastRanks = pastResults.mapValues { (_, sets) ->
            sets.map { Rank.determineRank(it.toSet(), actualAnswer) }
        }

        // 상태 업데이트
        AppState.championRank = findBestModel(pastRanks, SelectionStrategy.BEST_RANK_FIRST)
        AppState.championWins = findBestModel(pastRanks, SelectionStrategy.MOST_WINS_FIRST)
        AppState.pastTestReportHtml = generatePastReportHtml(pastResults, pastRanks, actualAnswer, AppState.championRank, AppState.championWins)

        // 미래 예측기 재훈련
        println("\n미래 예측용 모델 재훈련")
        val feFuture = FeatureEngineer()
        val tdFuture = feFuture.createTrainingData(fullHistory)

        val champions = setOf(AppState.championRank.modelName, AppState.championWins.modelName)
        val futureBaseModels = modelsToTrain.filterKeys { it in champions }
        val (futureTrainedModels, futureHeader) = trainAllModels(tdFuture, futureBaseModels)

        // 상태 업데이트 (예측기 & 데이터)
        AppState.bestPredictor_RankStrategy = LottoPredictor(futureTrainedModels[AppState.championRank.modelName]!!, futureHeader, feFuture)
        AppState.bestPredictor_WinsStrategy = LottoPredictor(futureTrainedModels[AppState.championWins.modelName]!!, futureHeader, feFuture)
        AppState.fullHistoryForPredict = fullHistory
        AppState.latestDrawsShortForPredict = fullHistory.takeLast(10)
        AppState.latestDrawsMidForPredict = fullHistory.takeLast(25)
        AppState.isInitialized = true

        println("업데이트 완료 (최신: ${AppState.lastDrawNo}회)")
    }

    // 내부 헬퍼 함수: 모든 모델 훈련
    private fun trainAllModels(data: DataFrame<*>, models: Map<String, Classifier>): Pair<Map<String, Map<Int, Classifier>>, Instances> {
        val trainer = LottoModelTrainer()
        val trained = mutableMapOf<String, Map<Int, Classifier>>()

        // 첫 모델 훈련하여 헤더 확보
        val firstKey = models.keys.first()
        trainer.train(data, models[firstKey]!!)
        trained[firstKey] = trainer.getModels()
        val header = trainer.dataHeader

        models.keys.drop(1).forEach {
            trainer.train(data, models[it]!!)
            trained[it] = trainer.getModels()
        }
        return Pair(trained, header)
    }

    // 내부 헬퍼 함수: 챔피언 선발
    private fun findBestModel(ranksMap: Map<String, List<Rank>>, strategy: SelectionStrategy): ModelScore {
        val scores = ranksMap.map { (name, list) ->
            val wins = list.filter { it != Rank.NONE }
            ModelScore(name, wins.minByOrNull { it.ordinal } ?: Rank.NONE, wins.size)
        }
        return scores.sortedWith(when(strategy) {
            SelectionStrategy.BEST_RANK_FIRST -> compareBy<ModelScore> { it.bestRank.ordinal }.thenByDescending { it.totalWins }
            SelectionStrategy.MOST_WINS_FIRST -> compareByDescending<ModelScore> { it.totalWins }.thenBy { it.bestRank.ordinal }
        }).first()
    }
}