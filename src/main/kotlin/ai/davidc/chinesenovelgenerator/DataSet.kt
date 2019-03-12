package ai.davidc.chinesenovelgenerator

import org.apache.commons.io.IOUtils
import java.io.FileInputStream


data class DataSet(val filePath: String) {
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
