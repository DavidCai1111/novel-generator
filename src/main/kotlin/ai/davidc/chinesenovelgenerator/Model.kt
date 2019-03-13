package ai.davidc.chinesenovelgenerator

import org.apache.commons.logging.LogFactory
import org.deeplearning4j.zoo.model.TextGenerationLSTM
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.springframework.stereotype.Component

@Component
class Model {
    private val logger = LogFactory.getLog(Model::class.java)
    val dataSetInfo = DataSet("./src/main/resources/data/tinyshakespeare.txt")

    private val model = TextGenerationLSTM
            .builder()
            .inputShape(intArrayOf(1, dataSetInfo.validCharacters.length))
            .totalUniqueCharacters(dataSetInfo.validCharacters.length)
            .updater(Adam())
            .build()

    private var inputArrayCache: INDArray? = null
    private var labelArrayCache: INDArray? = null

    val inputArray: INDArray
        get() {
            if (inputArrayCache != null) {
                return inputArrayCache!!
            }

            inputArrayCache = Nd4j.zeros(1, dataSetInfo.validCharacters.length, dataSetInfo.dataString.length)

            for (i in 0..(dataSetInfo.dataString.length - 2)) {
                inputArrayCache!!.putScalar(intArrayOf(0, dataSetInfo.validCharacters.indexOf(dataSetInfo.dataString[i]), i), 1)
            }

            return inputArrayCache!!
        }

    val labelArray: INDArray
        get() {
            if (labelArrayCache != null) {
                return labelArrayCache!!
            }

            labelArrayCache = Nd4j.zeros(1, dataSetInfo.validCharacters.length, dataSetInfo.dataString.length)

            for (i in 0..(dataSetInfo.dataString.length - 2)) {
                labelArrayCache!!.putScalar(intArrayOf(0, dataSetInfo.validCharacters.indexOf(dataSetInfo.dataString[i + 1]), i), 1)
            }

            return labelArrayCache!!
        }

    fun generate() {
        logger.info(model.toString())
    }
}
