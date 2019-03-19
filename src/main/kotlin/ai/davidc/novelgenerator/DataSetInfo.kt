package ai.davidc.novelgenerator

import org.apache.commons.io.IOUtils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import java.io.FileInputStream

const val MAX_WORD_LENGTH = 40

data class DataSetInfo(val filePath: String) {
    val dataString = IOUtils.toString(FileInputStream(filePath), "UTF-8").toString().substring(0, 10000)
    val inputArrays: INDArray
    val labelArrays: INDArray
    var paddingArray: INDArray
    var validCharacters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890\"\n',.?;()[]{}:!-_ "
    private val inputArraysLength = (dataString.length - 1) / 3

    private val logger = LoggerFactory.getLogger(DataSetInfo::class.java)

    init {
        logger.info("Data length: ${dataString.length}")

        inputArrays = Nd4j.zeros(inputArraysLength, validCharacters.length, MAX_WORD_LENGTH)
        labelArrays = Nd4j.zeros(inputArraysLength, validCharacters.length, MAX_WORD_LENGTH)
        paddingArray = Nd4j.zeros(inputArraysLength, MAX_WORD_LENGTH)

        for (i in 0..(inputArraysLength - 1)) {
            paddingArray.putScalar(intArrayOf(i, MAX_WORD_LENGTH - 1), 1)
        }

        logger.info(inputArrays.shapeInfoToString())
        logger.info(labelArrays.shapeInfoToString())

        var index = 0
        for (i in 0..(dataString.length - MAX_WORD_LENGTH - 1) step 3) {
            index++
            val sentence = dataString.substring(i, i + MAX_WORD_LENGTH)

            for (j in 0..(MAX_WORD_LENGTH - 1)) {
                inputArrays.putScalar(intArrayOf(index, validCharacters.indexOf(sentence[j]), j), 1)
                labelArrays.putScalar(intArrayOf(index, validCharacters.indexOf(dataString[i + MAX_WORD_LENGTH]), j), 1)
            }
        }
    }

    fun getSentenceToINDArray(sentence: String): INDArray {
        val array = Nd4j.zeros(1, validCharacters.length, MAX_WORD_LENGTH)

        var length = sentence.length

        if (length > MAX_WORD_LENGTH) {
            length = MAX_WORD_LENGTH
        }

        for (i in 0..(length - 1)) {
            array.putScalar(intArrayOf(0, validCharacters.indexOf(sentence[i]), i), 1)
        }

        return array
    }

    fun indArrayToCharacter(array: INDArray): String {
        var maxPrediction = Double.MIN_VALUE
        var maxPredictionIndex = -1

        for (i in 0..(validCharacters.length - 1)) {
            if (maxPrediction < array.getDouble(0, i, 0)) {
                maxPrediction = array.getDouble(0, i, 0)
                maxPredictionIndex = i
            }
        }

        return validCharacters[maxPredictionIndex].toString()
    }
}
