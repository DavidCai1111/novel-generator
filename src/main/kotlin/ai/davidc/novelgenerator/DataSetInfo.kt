package ai.davidc.novelgenerator

import org.apache.commons.io.IOUtils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.FileInputStream


data class DataSetInfo(val filePath: String) {
    val dataString = IOUtils.toString(FileInputStream(filePath), "UTF-8").toString().substring(0, 5000)
    var inputArray: INDArray
    var labelArray: INDArray
    var validCharacters = ""

    init {
        for (character in dataString) {
            if (validCharacters.indexOf(character) == -1) {
                validCharacters += character
            }
        }

        inputArray = Nd4j.zeros(1, validCharacters.length, dataString.length - 1)
        labelArray = Nd4j.zeros(1, validCharacters.length, dataString.length - 1)

        for (i in 0..(dataString.length - 2)) {
            inputArray.putScalar(intArrayOf(0, validCharacters.indexOf(dataString[i]), i), 1)
            labelArray.putScalar(intArrayOf(0, validCharacters.indexOf(dataString[i + 1]), i), 1)
        }
    }
}
