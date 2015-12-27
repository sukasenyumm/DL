namespace LibDL.Interface
{
    public interface INeuron
    {
        void RandomWeights();
        double Compute(double[] input);
    }
}
