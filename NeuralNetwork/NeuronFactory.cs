namespace NeuralNetwork;

public static class NeuronFactory
{
    private static readonly Random Random = new();
    
    public static Neuron CreateNeuron(int weights)
    {
        return new Neuron(RandomWeights(weights), RandomWeight());
    }
    
    private static double[] RandomWeights(int length)
    {
        var weights = new double[length];

        for (var i = 0; i < length; i++)
        {
            weights[i] = RandomWeight();
        }

        return weights;
    }

    private static double RandomWeight()
    {
        return Random.NextDouble() * 2 - 1;
    } 
}