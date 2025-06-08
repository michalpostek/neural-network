namespace NeuralNetwork;

public class NeuralNetwork
{
    private readonly Neuron[][] _hiddenLayers;
    private readonly Neuron[] _outputLayer;
    private readonly double _learningRate;

    public NeuralNetwork(int inputsCount, int hiddenLayersCount, int hiddenLayerSize, int outputsCount, double minInitWeight, double maxInitWeight, double learningRate)
    {
        _hiddenLayers = InitHiddenLayers(inputsCount, hiddenLayersCount, hiddenLayerSize, minInitWeight, maxInitWeight);
        _outputLayer = InitOutputLayer(hiddenLayerSize, outputsCount, minInitWeight, maxInitWeight);
        _learningRate = learningRate;
    }

    public double[] GetCurrentOutput(double[] inputs)
    {
        var currentInputs = inputs.ToList();

        for (var i = 0; i < _hiddenLayers.Length; i++)
        {
            var hiddenLayerOutputs = new double[_hiddenLayers[i].Length];

            for (var j = 0; j < _hiddenLayers[i].Length; j++)
            {
                hiddenLayerOutputs[j] = _hiddenLayers[i][j].ActivateAndGetOutput(currentInputs.ToArray());
            }

            currentInputs = hiddenLayerOutputs.ToList();
        }

        return _outputLayer.Select(x => x.ActivateAndGetOutput(currentInputs.ToArray())).ToArray();
    }

    private static Neuron[] InitOutputLayer(int hiddenLayerSize, int outputsCount, double minInitWeight, double maxInitWeight)
    {
        var layer = new Neuron[outputsCount];

        for (var i = 0; i < outputsCount; i++)
        {
            layer[i] = NeuronFactory.CreateNeuron(hiddenLayerSize, minInitWeight, maxInitWeight);
        }
        
        return layer;
    }

    private static Neuron[][] InitHiddenLayers(int inputsCount, int hiddenLayersCount, int hiddenLayerSize, double minInitWeight, double maxInitWeight)
    {
        var hiddenLayers = new Neuron[hiddenLayersCount][];

        for (var i = 0; i < hiddenLayersCount; i++)
        {
            var layer = new Neuron[hiddenLayerSize];

            for (var j = 0; j < hiddenLayerSize; j++)
            {
                var inputs = i == 0 ? inputsCount : hiddenLayerSize;
                layer[j] = NeuronFactory.CreateNeuron(inputs, minInitWeight, maxInitWeight);
            }
            
            hiddenLayers[i] = layer;
        }
        
        return hiddenLayers;
    }
}