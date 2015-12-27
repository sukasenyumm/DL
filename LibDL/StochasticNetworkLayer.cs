using System;
using LibDL.Interface;
using LibDL.Utils;
using LibDL.ActivationFunction;

namespace LibDL
{
    [Serializable]
    public class StochasticNetworkLayer : UsualNetworkLayer
    {
        
        private new StochasticNeuron[] neurons;
       
        /// <summary>
        ///   Gets the layer's neurons.
        /// </summary>
        /// 
        public new StochasticNeuron[] Neurons
        {
            get { return neurons; }
        }

        /*
         * Siapkan deklarasi untuk mendapatkan output layer stokastik yang terbentuk dari neuron stokastik
         */ 
        private double[] outStochastic;
        public double[] OutStochastic
        {
            get { return outStochastic; }
        }

        public StochasticNetworkLayer(int neuronsCount, int inputsCount)
            : this(neuronsCount, inputsCount, new BernoulliDistribution(alpha: 1)) { }

        public StochasticNetworkLayer(int neuronsCount, int inputsCount,IActivationFunction function)
            : base(neuronsCount, inputsCount, function)
        {
            //minimal input dan neuron adalah 1
            this.inputsCount = Math.Max(1, inputsCount);
            this.neuronsCount = Math.Max(1, neuronsCount);

            neurons = new StochasticNeuron[neuronsCount];
            for (int i = 0; i < neurons.Length; i++)
                base.neurons[i] = this.neurons[i] = new StochasticNeuron(inputsCount, function);
        }

        public override double[] Compute(double[] input)
        {
            // variabel lokal mencegah konfilk mutlithreading
            double[] output = new double[neuronsCount];

            // hitung output setiap neuron
            for (int i = 0; i < neurons.Length; i++)
                output[i] = neurons[i].Compute(input);

            // assign output
            this.output = output;

            return output;
        }

        public override void Randomize()
        {
            //generate bobot random dari setiap neuron
            foreach (StochasticNeuron neuron in neurons)
                neuron.RandomWeights();
        }

        public override void SetActivationFunction(IActivationFunction function)
        {
            //set fungsi aktifasi untuk setiap neuron
            for (int i = 0; i < neurons.Length; i++)
            {
                ((StochasticNeuron)neurons[i]).ActivationFunction = function;
            }
        }

        /*
         * Hitung estimasi probabilitas vektor dari setiap layer, dan kembalikan nilai tersebut
         * 
         */ 
        public double[] Generate(double[] input)
        {
            double[] sample = new double[neuronsCount];
            double[] output = new double[neuronsCount];

            for (int i = 0; i < neurons.Length; i++)
            {
                sample[i] = neurons[i].Generate(input);
                output[i] = neurons[i].Output;
            }

            this.outStochastic = sample;
            this.output = output;

            return sample;
        }

        /*
         * Fungsi untuk mengkopi bobot dari layer yang lain dengan urutan terbalik. 
         * sangat berguna untuk update visible layer dari hidden layer atau sebaliknya.
         *  
         */ 
        public void CopyReversedWeightsFrom(StochasticNetworkLayer layer)
        {
            for (int i = 0; i < Neurons.Length; i++)
                for (int j = 0; j < inputsCount; j++)
                    this.Neurons[i].Weights[j] = layer.Neurons[j].Weights[i];
        }

    }
}
