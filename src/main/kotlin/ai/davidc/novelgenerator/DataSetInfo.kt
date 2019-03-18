package ai.davidc.novelgenerator

import org.apache.commons.io.IOUtils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import java.io.FileInputStream

const val INPUT_WORD_LENGTH = 5
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
        inputArray = Nd4j.zeros(INPUT_WORD_LENGTH, MAX_WORD_LENGTH, validCharacters.length)
        labelArray = Nd4j.zeros(MAX_WORD_LENGTH, validCharacters.length)

        val words = dataString.split(" ")
        val inputArraysLength = ((words.size - INPUT_WORD_LENGTH) / INPUT_WORD_LENGTH)

        inputArrays = Nd4j.zeros(inputArraysLength, INPUT_WORD_LENGTH, MAX_WORD_LENGTH, validCharacters.length)
        labelArrays = Nd4j.zeros(inputArraysLength, MAX_WORD_LENGTH, validCharacters.length)

        logger.info("Get ${words.size} words")
        logger.info(inputArrays.shapeInfoToString())
        logger.info(labelArrays.shapeInfoToString())

        for (i in 0..(inputArraysLength - 1)) {
            for (j in 0..(INPUT_WORD_LENGTH - 1)) {
                val wordINDArray = getWordToINDArray(words[i * INPUT_WORD_LENGTH + j])

                for (k in 0..(MAX_WORD_LENGTH - 1)) {
                    inputArrays.put(intArrayOf(i, j, k), wordINDArray.getScalar(k))
                }
            }

            for (k in 0..(MAX_WORD_LENGTH - 1)) {
                val labelWordArray = getWordToINDArray(words[i * 5 + 5])
                labelArrays.put(intArrayOf(i), labelWordArray.getScalar(k))
            }
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

    fun INDArrayToWord(array: INDArray): String {
        var word = ""

        for (i in 0..(MAX_WORD_LENGTH - 1)) {
            var maxPrediction = Double.MIN_VALUE
            var maxPredictionIndex = -1

            for (j in 0..(validCharacters.length - 1)) {
                if (maxPrediction < array.getDouble(i, j)) {
                    maxPrediction = array.getDouble(i, j)
                    maxPredictionIndex = j
                }
            }

            if (isLastCharacter(array, i)) {
                return word
            }

            word += validCharacters[maxPredictionIndex]
        }

        return word
    }

    private fun isLastCharacter (wordINDArray: INDArray, index: Int): Boolean {
        for (i in index..(MAX_WORD_LENGTH - 1)) {
            for (j in 0..(validCharacters.length - 1)) {
                if (wordINDArray.getDouble(i, j) != 0.toDouble()) {
                    return false
                }
            }
        }

        return true
    }
}
