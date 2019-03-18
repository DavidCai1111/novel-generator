package ai.davidc.novelgenerator

import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import org.nd4j.linalg.factory.Nd4j
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

    @Test
    fun testWordAndINDArrayConversion() {
        val array = Nd4j.zeros(1, dataSetInfo.validCharacters.length)
        array.putScalar(intArrayOf(0, 0), 1)

        val character = dataSetInfo.indArrayToCharacter(array)

        Assert.assertEquals("A", character)
    }
}
