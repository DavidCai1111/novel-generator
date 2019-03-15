package ai.davidc.novelgenerator

import org.apache.commons.io.IOUtils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.FileInputStream


data class DataSetInfo(val filePath: String) {
    val dataString = IOUtils.toString(FileInputStream(filePath), "UTF-8").toString().substring(0, 500)
    var inputArray: INDArray
    var labelArray: INDArray

    init {
        inputArray = Nd4j.zeros(1, VALID_CHARACTERS.length, dataString.length - 1)
        labelArray = Nd4j.zeros(1, VALID_CHARACTERS.length, dataString.length - 1)

        for (i in 0..(dataString.length - 2)) {
            inputArray.putScalar(intArrayOf(0, VALID_CHARACTERS.indexOf(dataString[i]), i), 1)
            labelArray.putScalar(intArrayOf(0, VALID_CHARACTERS.indexOf(dataString[i + 1]), i), 1)
        }
    }
}
