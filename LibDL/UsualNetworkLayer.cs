using System;
using LibDL.Interface;
using LibDL.Utils;
using LibDL.ActivationFunction;

namespace LibDL
{
    [Serializable]
    public class UsualNetworkLayer:NetworkLayer
    {

        private new UsualNeuron[] neurons;

        /// <summary>
        ///   Gets the layer's neurons.
        /// </summary>
        /// 
        public new UsualNeuron[] Neurons
        {
            get { return neurons; }
        }


        public UsualNetworkLayer(int neuronsCount, int inputsCount)
            : this(neuronsCount, inputsCount, new SigmoidFunction(alpha: 0.001)) { }

        public UsualNetworkLayer(int neuronsCount, int inputsCount, IActivationFunction function)
            : base(neuronsCount, inputsCount, function)
        {
            //minimal input dan neuron adalah 1 (KODE 1-1)
            this.inputsCount = Math.Max(1, inputsCount);
            this.neuronsCount = Math.Max(1, neuronsCount);

            //inisialisasi neuron sejumlah banyaknya neuron yang diinginkan(KODE 1-1)
            neurons = new UsualNeuron[this.neuronsCount];
            // buat setiap neuron pada layer
            for (int i = 0; i < neurons.Length; i++)
                base.neurons[i] = this.neurons[i] = new UsualNeuron(inputsCount, function);
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
            foreach (Neuron neuron in neurons)
                neuron.RandomWeights();
        }

        public override void SetActivationFunction(IActivationFunction function)
        {
            //set fungsi aktifasi untuk setiap neuron
            for (int i = 0; i < neurons.Length; i++)
            {
                ((UsualNeuron)neurons[i]).ActivationFunction = function;
            }
        }

    }
}

