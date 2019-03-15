package ai.davidc.novelgenerator

import org.apache.commons.logging.LogFactory
import org.springframework.boot.CommandLineRunner
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class NovelGeneratorApplication(val model: Model) : CommandLineRunner {
    private val logger = LogFactory.getLog(NovelGeneratorApplication::class.java)
    
    override fun run(vararg args: String?) {
//        model.load()
        model.train("./src/main/resources/data/data.txt", 1000)
//        logger.info("Generated novel: ${model.generate("A", 200)}")
    }
}

fun main(args: Array<String>) {
    runApplication<NovelGeneratorApplication>(*args)
}
