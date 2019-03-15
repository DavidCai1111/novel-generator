package ai.davidc.novelgenerator

import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.test.context.junit4.SpringRunner

@RunWith(SpringRunner::class)
@SpringBootTest
class DataSetInfoTests {
    private val dataSetInfo = DataSetInfo("./src/main/resources/data/data.txt")

    @Test
    fun testDataString() {
        Assert.assertNotEquals(dataSetInfo.dataString.length, 0)
    }

    @Test
    fun testDataCharacters() {
        val indexes = IntArray(dataSetInfo.dataString.length)

        for (i in 0..(dataSetInfo.dataString.length - 1)) {
            indexes[i] = dataSetInfo.validCharacters.indexOf(dataSetInfo.dataString[i])
        }

        for (i in indexes) {
            Assert.assertNotEquals(i, -1)
        }
    }
}
