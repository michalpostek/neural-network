namespace NeuralNetwork;

public static class NeuronFactory
{
    private static readonly Random Random = new();
    
    public static Neuron CreateNeuron(int weights, double min, double max)
    {
        return new Neuron(RandomWeights(weights, min, max), RandomWeight(min, max));
    }
    
    private static double[] RandomWeights(int length, double min, double max)
    {
        var weights = new double[length];

        for (var i = 0; i < length; i++)
        {
            weights[i] = RandomWeight(min, max);
        }

        return weights;
    }

    private static double RandomWeight(double min, double max)
    {
        return Random.NextDouble() * (max - min) + min;
    } 
}