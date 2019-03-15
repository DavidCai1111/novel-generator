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
import org.nd4j.linalg.schedule.ExponentialSchedule
import org.nd4j.linalg.schedule.ScheduleType
import org.springframework.stereotype.Component
import java.io.File

const val VALID_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890\"\n',.?;()[]{}:!- _"

@Component
class Model {
    private val logger = LogFactory.getLog(Model::class.java)

    private var model: MultiLayerNetwork = MultiLayerNetwork(NeuralNetConfiguration
            .Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .miniBatch(true)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
            .updater(Adam(ExponentialSchedule(ScheduleType.ITERATION, 0.01, 0.99)))
            .list()
            .layer(0, LSTM
                    .Builder()
                    .nIn(VALID_CHARACTERS.length)
                    .nOut(256)
                    .dropOut(Dropout(0.2))
                    .activation(Activation.TANH)
                    .build()
            )
            .layer(1, LSTM
                    .Builder()
                    .nOut(256)
                    .dropOut(Dropout(0.2))
                    .activation(Activation.TANH)
                    .build()
            )
            .layer(2, RnnOutputLayer
                    .Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.SOFTMAX)
                    .nOut(VALID_CHARACTERS.length)
                    .build()
            )
            .backprop(true)
            .pretrain(false)
            .build())

    private val modelFile = File("./src/main/resources/model")

    init {
        model.init()
    }

    fun train(txtPath: String, epoch: Int = 1) {
        val dataSetInfo = DataSetInfo(txtPath)

        logger.info("InputArray Shape: ${dataSetInfo.inputArray.shapeInfoToString()}")
        logger.info("LabelArray Shape: ${dataSetInfo.labelArray.shapeInfoToString()}")

        model.setListeners(ScoreIterationListener(10))
        for (i in 0..epoch) {
            if (i % 50 == 0) {
                logger.info(generate("A", 200))
            }
            model.fit(dataSetInfo.inputArray, dataSetInfo.labelArray)
        }

        model.save(modelFile)
    }

    fun load() {
        model = ModelSerializer.restoreMultiLayerNetwork(modelFile)
    }

    fun generate(firstCharacter: String, length: Int): String {
        var inputArray = Nd4j.zeros(VALID_CHARACTERS.length)
        inputArray.putScalar(0, 1)

        model.rnnClearPreviousState()

        var output = "A"

        for (i in 0..(length - 1)) {
            val outputArray = model.rnnTimeStep(inputArray)

            var maxPrediction = Double.MIN_VALUE
            var maxPredictionIndex = -1

            for (j in 0..(VALID_CHARACTERS.length - 1)) {
                if (maxPrediction < outputArray.getDouble(j)) {
                    maxPrediction = outputArray.getDouble(j)
                    maxPredictionIndex = j
                }
            }

            if (maxPredictionIndex == -1) {
                logger.error("maxPredictionIndex == -1")
            }

            output += VALID_CHARACTERS[maxPredictionIndex]

            inputArray = Nd4j.zeros(VALID_CHARACTERS.length)
            inputArray.putScalar(maxPredictionIndex.toLong(), 1)
        }

        return output
    }
}
