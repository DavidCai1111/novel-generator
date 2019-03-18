package ai.davidc.novelgenerator

import org.apache.commons.io.IOUtils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import java.io.FileInputStream

const val MAX_WORD_LENGTH = 40

data class DataSetInfo(val filePath: String) {
    private val logger = LoggerFactory.getLogger(DataSetInfo::class.java)
    val dataString = IOUtils.toString(FileInputStream(filePath), "UTF-8").toString().substring(0, 10000)
    val inputArrays: INDArray
    val labelArrays: INDArray
    var inputArray: INDArray
    var labelArray: INDArray
    var validCharacters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890\"\n',.?;()[]{}:!-_ "

    init {
        inputArray = Nd4j.zeros(MAX_WORD_LENGTH, validCharacters.length)
        labelArray = Nd4j.zeros(validCharacters.length)

        val inputArraysLength = (dataString.length - 1) / MAX_WORD_LENGTH

        inputArrays = Nd4j.zeros(inputArraysLength, MAX_WORD_LENGTH, validCharacters.length)
        labelArrays = Nd4j.zeros(inputArraysLength, validCharacters.length)

        logger.info(inputArrays.shapeInfoToString())
        logger.info(labelArrays.shapeInfoToString())

        for (i in 0..(dataString.length - 1) step MAX_WORD_LENGTH) {
            inputArrays.put(intArrayOf(i), getWordToINDArray(dataString.substring(i, i + MAX_WORD_LENGTH)))
            labelArrays.putScalar(intArrayOf(i, validCharacters.indexOf(dataString[i + MAX_WORD_LENGTH])), 1)
        }
    }

    fun getWordToINDArray (word: String): INDArray {
        val array = Nd4j.zeros(MAX_WORD_LENGTH, validCharacters.length)

        var length = word.length

        if (length > MAX_WORD_LENGTH) {
            length = MAX_WORD_LENGTH
        }

        for (i in 0..(length - 1)) {
            array.putScalar(intArrayOf(i, validCharacters.indexOf(word[i])), 1)
        }

        return array
    }

    fun indArrayToCharacter(array: INDArray): String {
        var maxPrediction = Double.MIN_VALUE
        var maxPredictionIndex = -1

        for (i in 0..(validCharacters.length - 1)) {
            if (maxPrediction < array.getDouble(i)) {
                maxPrediction = array.getDouble(i)
                maxPredictionIndex = i
            }
        }

        return validCharacters[maxPredictionIndex].toString()
    }
}
