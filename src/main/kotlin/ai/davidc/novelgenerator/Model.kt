package ai.davidc.novelgenerator

import org.apache.commons.logging.LogFactory
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.dropout.Dropout
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.springframework.stereotype.Component
import java.io.File

@Component
class Model {
    private val logger = LogFactory.getLog(Model::class.java)

    val dataSetInfo = DataSetInfo("./src/main/resources/data/data.txt")

    private var model: MultiLayerNetwork = MultiLayerNetwork(NeuralNetConfiguration
            .Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
            .miniBatch(true)
            .l2(0.001)
            .updater(Adam())
            .list()
            .layer(0, LSTM
                    .Builder()
                    .nIn(dataSetInfo.validCharacters.length)
                    .nOut(30)
                    .dropOut(Dropout(0.2))
                    .activation(Activation.TANH)
                    .build()
            )
            .layer(1, LSTM
                    .Builder()
                    .nOut(30)
                    .dropOut(Dropout(0.2))
                    .activation(Activation.TANH)
                    .build()
            )
            .layer(2, RnnOutputLayer
                    .Builder(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX)
                    .nOut(dataSetInfo.validCharacters.length)
                    .build()
            )
            .backprop(true)
            .build()
    )

    private val modelFile = File("./src/main/resources/model")

    init {
        model.init()
    }

    fun train(epoch: Int = 1) {
        logger.info("Valid characters: ${dataSetInfo.validCharacters}")
        logger.info("InputArray Shape: ${dataSetInfo.inputArray.shapeInfoToString()}")
        logger.info("LabelArray Shape: ${dataSetInfo.labelArray.shapeInfoToString()}")

        model.setListeners(ScoreIterationListener(10))

        for (i in 0..epoch) {
            model.fit(dataSetInfo.inputArray, dataSetInfo.labelArray)

            if (i % 50 == 0) {
                logger.info(generate("A", 200))
            }
        }

        model.save(modelFile)
    }

    fun load() {
        model = ModelSerializer.restoreMultiLayerNetwork(modelFile)
    }

    fun generate(firstCharacter: String, length: Int): String {
        var inputArray = Nd4j.zeros(dataSetInfo.validCharacters.length)
        inputArray.putScalar(0, dataSetInfo.validCharacters.indexOf(firstCharacter))

        model.rnnClearPreviousState()

        var output = firstCharacter

        for (i in 0..(length - 1)) {
            val outputArray = model.rnnTimeStep(inputArray)

            var maxPrediction = Double.MIN_VALUE
            var maxPredictionIndex = -1

            for (j in 0..(dataSetInfo.validCharacters.length - 1)) {
                if (maxPrediction < outputArray.getDouble(j)) {
                    maxPrediction = outputArray.getDouble(j)
                    maxPredictionIndex = j
                }
            }

            if (maxPredictionIndex == -1) {
                logger.error("maxPredictionIndex == -1")
            }

            output += dataSetInfo.validCharacters[maxPredictionIndex]

            inputArray = Nd4j.zeros(dataSetInfo.validCharacters.length)
            inputArray.putScalar(maxPredictionIndex.toLong(), 1)
        }

        return output
    }
}
