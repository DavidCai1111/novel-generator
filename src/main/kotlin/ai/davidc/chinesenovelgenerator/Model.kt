package ai.davidc.chinesenovelgenerator

import org.apache.commons.io.IOUtils
import org.apache.commons.logging.LogFactory
import org.deeplearning4j.zoo.model.TextGenerationLSTM
import org.springframework.stereotype.Component
import java.io.FileInputStream

@Component
class Model {
    private val logger = LogFactory.getLog(Model::class.java)
    private val model = TextGenerationLSTM
            .builder()
            .build()

    fun generate() {
        logger.info(model.toString())
    }

    fun getDataString(): String {
        return DataSetInfo("./src/main/resources/data/besieged-city.txt").dataString
    }

    fun getValidChars(): String {
        return DataSetInfo("./src/main/resources/data/besieged-city.txt").validCharacters
    }
}

data class DataSetInfo(val filePath: String) {
    val dataString = IOUtils.toString(FileInputStream(filePath), "UTF-8").replace("\n", "")

    val validCharacters: String
        get() {
            var validCharacters = ""

            for (character in dataString) {
                if (validCharacters.indexOf(character) < 0) {
                    validCharacters += character
                }
            }

            return validCharacters
        }
}
