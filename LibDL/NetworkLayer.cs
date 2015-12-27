using System;
using LibDL.Interface;
using LibDL.Utils;

namespace LibDL
{
    [Serializable]
    public abstract class NetworkLayer
    {
        /*
         * Jumlah input data
         */
        protected int inputsCount = 0;
        public int InputsCount
        {
            get { return inputsCount; }
            set { inputsCount = value; }
        }
        /*
         * Nilai output vektor dari layer
         */
        protected double[] output;
        public double[] Output
        {
            get { return output; }
        }

        /*
         * Deklarasi beberapa activation neuron pada suatu layer
         */
        protected int neuronsCount = 0;
        protected Neuron[] neurons;

        public Neuron[] Neurons
        {
            get { return neurons; }
        }

        public NetworkLayer(int neuronsCount, int inputsCount, IActivationFunction function)
        {
            //minimal input dan neuron adalah 1
            this.inputsCount = Math.Max(1, inputsCount);
            this.neuronsCount = Math.Max(1, neuronsCount);

            //inisialisasi neuron sejumlah banyaknya neuron yang diinginkan
            neurons = new Neuron[this.neuronsCount];
            // buat setiap neuron pada layer
            for (int i = 0; i < neurons.Length; i++)
                neurons[i] = new Neuron(inputsCount, function);
        }

        public virtual double[] Compute(double[] input)
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

        //(KODE 1-2)
        public virtual void Randomize()
        {
            //generate bobot random dari setiap neuron
            foreach ( Neuron neuron in neurons )
                neuron.RandomWeights( );
        }

        public virtual void SetActivationFunction(IActivationFunction function)
        {
            //set fungsi aktifasi untuk setiap neuron
            for (int i = 0; i < neurons.Length; i++)
            {
                ((Neuron)neurons[i]).ActivationFunction = function;
            }
        }
    }
}
