package ai.davidc.chinesenovelgenerator

import org.apache.commons.logging.LogFactory
import org.springframework.boot.CommandLineRunner
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class ChineseNovelGeneratorApplication(val model: Model) : CommandLineRunner {
    private val logger = LogFactory.getLog(ChineseNovelGeneratorApplication::class.java)
    
    override fun run(vararg args: String?) {
        model.load()
//        model.train("./src/main/resources/data/data.txt", 1000)
        logger.info("Generated novel: ${model.generate("A", 200)}")
    }
}

fun main(args: Array<String>) {
    runApplication<ChineseNovelGeneratorApplication>(*args)
}
