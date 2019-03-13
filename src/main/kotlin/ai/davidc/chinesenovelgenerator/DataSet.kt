package ai.davidc.chinesenovelgenerator

import org.apache.commons.io.IOUtils
import java.io.FileInputStream


data class DataSet(val filePath: String) {
    val dataString = IOUtils.toString(FileInputStream(filePath), "UTF-8")

    val validCharacters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890\\\"\\n',.?;()[]{}:!- "
}
