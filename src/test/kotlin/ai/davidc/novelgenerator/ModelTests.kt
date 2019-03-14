package ai.davidc.novelgenerator

import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.test.context.junit4.SpringRunner

@RunWith(SpringRunner::class)
@SpringBootTest
class ModelTests {
    val dataSetInfo = DataSetInfo("./src/main/resources/data/data.txt")

    @Test
    fun testDataString() {
        Assert.assertNotEquals(dataSetInfo.dataString.length, 0)
    }
}
