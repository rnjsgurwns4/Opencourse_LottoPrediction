// src/main/kotlin/lotto/LottoModelTrainer.kt
package lotto

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.*
import weka.classifiers.Classifier
import weka.classifiers.functions.Logistic // 예: 로지스틱 회귀 모델 (간단하고 빠름)
import weka.core.Attribute
import weka.core.DenseInstance
import weka.core.Instances
import org.jetbrains.kotlinx.dataframe.DataRow // ★ import 추가
import org.jetbrains.kotlinx.dataframe.api.rows


/**
 * 1번부터 45번까지, 총 45개의 개별 예측 모델을 훈련시키는 클래스
 */
class LottoModelTrainer {

    // 1번~45번 모델을 저장할 맵
    private val models = mutableMapOf<Int, Classifier>()

    // Weka가 데이터 구조를 인식할 수 있도록 하는 '헤더' 정보
    // (이 헤더는 45개 모델 모두에게 공통적으로 사용됨)
    lateinit var dataHeader: Instances
        private set // 외부에서는 읽기만 가능

    /**
     * 훈련 데이터셋(DataFrame)을 Weka의 Instances 객체로 변환합니다.
     * 이 과정에서 dataHeader가 생성됩니다.
     */
    private fun convertDataFrameToInstances(data: DataFrame<*>): Instances {
        val attributes = ArrayList<Attribute>()

        // 특성(Attribute) 정의
        // 1. recency (숫자형)
        attributes.add(Attribute("recency"))
        // 2. freq_latest (숫자형)
        attributes.add(Attribute("freq_latest"))
        // 3. freq_total (숫자형)
        attributes.add(Attribute("freq_total"))

        // 4. 정답(label) (범주형: "true", "false")
        val labelValues = arrayListOf("true", "false")
        attributes.add(Attribute("label", labelValues))

        // 1. 데이터 헤더(구조) 생성
        val instances = Instances("LottoFeatures", attributes, 0)
        // 정답(label)이 몇 번째 속성인지 Weka에 알려줌 (마지막)
        instances.setClassIndex(attributes.size - 1)

        // 2. 데이터 채우기 (DataFrame -> Instances)
        data.rows().forEach { row ->
            val instance = DenseInstance(attributes.size)
            instance.setDataset(instances) // 헤더와 연결
            instance.setValue(attributes[0], row.getValue<Int>("recency").toDouble())
            instance.setValue(attributes[1], row.getValue<Int>("freq_latest").toDouble())
            instance.setValue(attributes[2], row.getValue<Int>("freq_total").toDouble())
            instance.setValue(attributes[3], row.getValue<String>("label"))
            instances.add(instance)
        }

        this.dataHeader = Instances(instances, 0) // 헤더 정보만 복사해서 저장
        return instances
    }

    /**
     * 45개 모델을 훈련시킵니다.
     * @param trainingData FeatureEngineer가 생성한 DataFrame
     */
    fun train(trainingData: DataFrame<*>) {
        println("[LottoModelTrainer] 45개 모델 훈련 시작 (수동 GroupBy 방식으로 변경)...")
        println(trainingData.columnNames())
        // ★ 1. "number" 열을 List<Int>로 미리 추출 (Materialize)
        //    (만약 여기서 'Column not found'가 난다면,
        //     FeatureEngineer -> Main.kt -> LottoModelTrainer로 오는
        //     데이터 전달 과정 자체에 문제가 있는 것입니다.)
        println("[LottoModelTrainer] 'number' 열을 List로 추출 시도...")
        val numberColumn: List<Int> = try {
            // "number" 열에 접근해서 List<Int>로 변환
            trainingData["number"].toList() as List<Int>
        } catch (e: Exception) {
            println("\n!!!!!! [오류] 'number' 열을 List로 변환 중 예외 발생 !!!!!!")
            println("trainingData['number'].toList() 실패. 'number' 열을 찾을 수 없습니다.")
            println("예외 메시지: ${e.message}\n")
            throw e
        }
        println("[LottoModelTrainer] 'number' 열 List 추출 성공. (size=${numberColumn.size})")

        // ★ 2. 1~45번 그룹(Map)을 만들고, 'number' 값에 따라 모든 행(row)을 재분배
        val dataByNumber = mutableMapOf<Int, MutableList<DataRow<*>>>()
        for (num in 1..45) {
            dataByNumber[num] = mutableListOf() // 빈 리스트로 초기화
        }

        println("[LottoModelTrainer] 1..45번 그룹으로 데이터 재분배 시작...")
        // trainingData의 모든 행(rows)과 위에서 뽑은 numberColumn 리스트를 'zip' (1:1 매칭)
        trainingData.rows().zip(numberColumn).forEach { (row, numberValue) ->
            // numberValue(예: 5)에 해당하는 리스트에 'row'를 추가
            dataByNumber[numberValue]?.add(row)
        }
        println("[LottoModelTrainer] 데이터 재분배 완료.")


        // ★ 3. 1..45번 루프 (기존 'groupBy'와 동일한 효과)
        for (num in 1..45) {
            try {
                // filter 대신, 이미 분배된 리스트를 가져옴
                val numDataRows: List<DataRow<*>> = dataByNumber[num]!!

                if (numDataRows.isEmpty()) {
                    println("[LottoModelTrainer] ... $num 번 훈련 데이터가 없습니다. 건너뜁니다.")
                    continue
                }

                // ★ 4. List<DataRow>를 다시 DataFrame으로 변환
                //    (convertDataFrameToInstances가 DataFrame을 받기 때문)
                val numData = numDataRows.toDataFrame()

                val trainingInstances = convertDataFrameToInstances(numData)
                val classifier = Logistic()
                classifier.buildClassifier(trainingInstances)

                models[num] = classifier

                if (num % 5 == 0 || num == 1 || num == 45) {
                    println("[LottoModelTrainer] ... $num 번 모델 훈련 완료 ...")
                }
            } catch (e: Exception) {
                println("\n!!!!!! [오류] $num 번 모델 훈련 중 예외 발생 !!!!!!")
                println("예외 메시지: ${e.message}\n")
                throw e
            }
        }
        println("[LottoModelTrainer] 총 45개 모델 훈련 완료.")
    }

    fun getModels(): Map<Int, Classifier> = models
}