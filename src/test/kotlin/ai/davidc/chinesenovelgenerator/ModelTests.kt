package ai.davidc.chinesenovelgenerator

import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.test.context.junit4.SpringRunner

@RunWith(SpringRunner::class)
@SpringBootTest
class ModelTests {
    @Autowired
    lateinit var model: Model

    @Test
    fun testGetDataString() {
        val dataString = model.dataSetInfo.dataString

        Assert.assertNotEquals(dataString.length, 0)
    }

    @Test
    fun testGetValidCharacters() {
        val validChars = model.dataSetInfo.validCharacters

        Assert.assertNotEquals(validChars.length, 0)
    }
}
