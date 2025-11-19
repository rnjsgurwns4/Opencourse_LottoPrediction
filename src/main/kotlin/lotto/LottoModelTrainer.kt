package lotto

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.*
import weka.classifiers.Classifier
import weka.core.Attribute
import weka.core.DenseInstance
import weka.core.Instances
import org.jetbrains.kotlinx.dataframe.DataRow
import org.jetbrains.kotlinx.dataframe.api.rows
import weka.classifiers.AbstractClassifier


// 1~45번 숫자에 대한 분석
class LottoModelTrainer {

    private val models = mutableMapOf<Int, Classifier>()

    lateinit var dataHeader: Instances
        private set

    // dataHeader 생성
    private fun convertDataFrameToInstances(data: DataFrame<*>): Instances {
        val attributes = ArrayList<Attribute>()

        attributes.add(Attribute("recency"))
        attributes.add(Attribute("freq_short"))
        attributes.add(Attribute("freq_mid"))
        attributes.add(Attribute("freq_total_main"))
        attributes.add(Attribute("freq_total_bonus"))

        val labelValues = arrayListOf("true", "false")
        attributes.add(Attribute("label", labelValues))

        // 데이터 헤더(구조) 생성
        val instances = Instances("LottoFeatures", attributes, 0)
        instances.setClassIndex(attributes.size - 1)

        // 데이터 채우기 (DataFrame -> Instances)
        data.rows().forEach { row ->
            val instance = DenseInstance(attributes.size)
            instance.setDataset(instances)
            instance.setValue(attributes[0], row.getValue<Int>("recency").toDouble())
            instance.setValue(attributes[1], row.getValue<Int>("freq_short").toDouble())
            instance.setValue(attributes[2], row.getValue<Int>("freq_mid").toDouble())
            instance.setValue(attributes[3], row.getValue<Int>("freq_total_main").toDouble())
            instance.setValue(attributes[4], row.getValue<Int>("freq_total_bonus").toDouble())
            instance.setValue(attributes[5], row.getValue<String>("label"))
            instances.add(instance)
        }

        this.dataHeader = Instances(instances, 0)
        return instances
    }

    // 훈련
    fun train(trainingData: DataFrame<*>, baseClassifier: Classifier) {
        println("45개 모델 훈련 시작")
        val modelName = baseClassifier.javaClass.simpleName

        models.clear()


        val numberColumn: List<Int> = try {
            trainingData["number"].toList() as List<Int>
        } catch (e: Exception) {
            println("예외 메시지: ${e.message}\n")
            throw e
        }

        // 1~45번 Map 생성
        val dataByNumber = mutableMapOf<Int, MutableList<DataRow<*>>>()
        for (num in 1..45) {
            dataByNumber[num] = mutableListOf() // 빈 리스트로 초기화
        }

        // trainingData의 모든 행과 numberColumn 리스트를  매칭
        trainingData.rows().zip(numberColumn).forEach { (row, numberValue) ->
            dataByNumber[numberValue]?.add(row)
        }

        // 1..45번 루프
        for (num in 1..45) {
            try {
                val numDataRows: List<DataRow<*>> = dataByNumber[num]!!

                if (numDataRows.isEmpty()) {
                    println("[LottoModelTrainer] ... $num 번 훈련 데이터가 없습니다. 건너뜁니다.")
                    continue
                }

                // List<DataRow>를 다시 DataFrame으로 변환
                val numData = numDataRows.toDataFrame()

                val trainingInstances = convertDataFrameToInstances(numData)
                val classifier = AbstractClassifier.makeCopy(baseClassifier)
                classifier.buildClassifier(trainingInstances)

                models[num] = classifier

            } catch (e: Exception) {
                println("예외 메시지: ${e.message}\n")
                throw e
            }
        }
        println("총 45개 모델 훈련 완료.")
    }

    fun getModels(): Map<Int, Classifier> {
        return models.toMap()
    }
}