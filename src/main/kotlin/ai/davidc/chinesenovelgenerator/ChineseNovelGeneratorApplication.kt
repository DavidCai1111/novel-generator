package ai.davidc.chinesenovelgenerator

import org.springframework.boot.CommandLineRunner
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class ChineseNovelGeneratorApplication: CommandLineRunner {
    override fun run(vararg args: String?) {

    }
}

fun main(args: Array<String>) {
    runApplication<ChineseNovelGeneratorApplication>(*args)
}
