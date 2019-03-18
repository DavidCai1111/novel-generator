package ai.davidc.novelgenerator

import org.apache.commons.logging.LogFactory
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.springframework.stereotype.Component
import java.io.File

@Component
class Model {
    private val logger = LogFactory.getLog(Model::class.java)

    private val dataSetInfo = DataSetInfo("./src/main/resources/data/data.txt")

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
                    .nIn(MAX_WORD_LENGTH)
                    .nOut(30)
                    .activation(Activation.TANH)
                    .build()
            )
            .layer(1, RnnOutputLayer
                    .Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.SOFTMAX)
                    .nOut(dataSetInfo.validCharacters.length)
                    .build()
            )
            .pretrain(false)
            .backprop(true)
            .build()
    )

    private val modelFile = File("./src/main/resources/model")

    init {
        model.init()
    }

    fun train(epoch: Int = 1) {
        model.setListeners(ScoreIterationListener(10))

        for (i in 0..epoch) {
            model.fit(dataSetInfo.inputArrays, dataSetInfo.labelArrays)

            if (i % 10 == 0) {
                logger.info(generate("We are accounted poor citizens, the city", 200))
            }
        }

        model.save(modelFile)
    }

    fun load() {
        model = ModelSerializer.restoreMultiLayerNetwork(modelFile)
    }

    fun generate(firstSentence: String, length: Int): String {
        val inputArray = dataSetInfo.getSentenceToINDArray(firstSentence)

        model.rnnClearPreviousState()

        var output = firstSentence

        for (i in 0..(length - 1)) {
            val outputArray = model.rnnTimeStep(inputArray)

            output += dataSetInfo.indArrayToCharacter(outputArray)
        }

        return output
    }
}
