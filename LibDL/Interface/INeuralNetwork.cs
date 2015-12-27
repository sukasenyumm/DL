using System.Collections.Generic;
namespace LibDL.Interface
{
    public interface INeuralNetwork
    {
        double[] Compute(double[] input);
        void Randomize();
        double[] ZeroOutput(double[] output);
        void setAllErrCost(List<double> allcosterr);
        List<double> getAllErrCost();
        void setTimeLearn(string time);
        string getTimeLearn();
    }
}
