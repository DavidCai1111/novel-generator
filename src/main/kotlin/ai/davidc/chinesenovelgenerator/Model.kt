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
        var input = IOUtils.toString(FileInputStream("./src/main/resources/data/besieged-city.txt"), "UTF-8")

        input = input.replace("\n", "")

        return input
    }
}
