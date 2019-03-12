package ai.davidc.chinesenovelgenerator

import org.apache.commons.logging.LogFactory
import org.deeplearning4j.zoo.model.TextGenerationLSTM
import org.springframework.stereotype.Component

@Component
class Model {
    private val logger = LogFactory.getLog(Model::class.java)
    private val model = TextGenerationLSTM
            .builder()
            .build()

    val dataSetInfo = DataSet("./src/main/resources/data/besieged-city.txt")

    fun generate() {
        logger.info(model.toString())
    }
}
