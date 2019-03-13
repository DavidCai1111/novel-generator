package ai.davidc.chinesenovelgenerator

import org.apache.commons.logging.LogFactory
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.springframework.stereotype.Component

const val VALID_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890\\\"\\n',.?;()[]{}:!-"

@Component
class Model {
    private val logger = LogFactory.getLog(Model::class.java)

    private val model: MultiLayerNetwork = MultiLayerNetwork(NeuralNetConfiguration
            .Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .cacheMode(CacheMode.NONE)
            .updater(Updater.RMSPROP)
            .trainingWorkspaceMode(WorkspaceMode.ENABLED)
            .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
            .list()
            .layer(0, GravesLSTM
                    .Builder()
                    .nIn(VALID_CHARACTERS.length)
                    .nOut(256)
                    .activation(Activation.TANH)
                    .build()
            )
            .layer(1, GravesLSTM
                    .Builder()
                    .nOut(256)
                    .activation(Activation.TANH)
                    .build()
            )
            .layer(2, RnnOutputLayer
                    .Builder(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX)
                    .nOut(VALID_CHARACTERS.length)
                    .build()
            )
            .backpropType(BackpropType.TruncatedBPTT)
            .tBPTTForwardLength(50)
            .tBPTTBackwardLength(50)
            .build())

    init {
        model.init()
    }

    fun train(txtPath: String) {
        val dataSetInfo = DataSetInfo(txtPath)

        model.addListeners(ScoreIterationListener(5))
        model.fit(dataSetInfo.inputArray, dataSetInfo.labelArray)
    }
}
