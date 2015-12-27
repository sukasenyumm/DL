using System;
using LibDL.Interface;
using LibDL.Utils;

namespace LibDL
{
    [Serializable]
    public class Neuron : INeuron
    {  
        /*
         * Jumlah input unit neuron
         */
        protected int inputsCount = 0;
        public int InputsCount
        {
            get { return inputsCount; }
            set { inputsCount = value; }
        }

        /*
         * Bobot setiap neuron
         */
        protected double[] weights = null;
        public double[] Weights
        {
            get { return weights; }
            set { weights = value; }
        }
        /*
         * Nilai output neuron
         */
        protected double output = 0;
        public double Output
        {
            get { return output; }
        }

        /*
         * Deklarasi fungsi aktifasi
         */
        protected IActivationFunction function = null;
        public IActivationFunction ActivationFunction
        {
            get { return function; }
            set { function = value; }
        }

        /*
         * Nilai ambang batas sebelum dimasukkan kedalam fungsi aktifasi atau bias
         */
        protected double threshold = 0.0;
        public double Threshold
        {
            get { return threshold; }
            set { threshold = value; }
        }
        
        /*
         * Random variable
         */
        protected static ThreadSafeRandom rnd = new ThreadSafeRandom();

        /*
         * Inisialiasi neuron dari input yang masuk
         */
        protected Neuron(int inputs)
        {
            // allocate weights
            inputsCount = Math.Max(1, inputs);
            weights = new double[inputsCount];

            // randomize the neuron
            RandomWeights();
        }
        public Neuron(int inputs,IActivationFunction function)
        {
            //alokasikan bobot setiap neuron
            inputsCount = Math.Max(1, inputs);
            weights = new double[inputsCount];

            //generate nilai random  pada setiap neuron
            RandomWeights();

            //inisialisasi fungsi aktivasi
            this.function = function;
        }

        /*
         * Generate nilai random pada setiap bobot neuron (KODE 1-2)
         */
        public virtual void RandomWeights()
        {
            for (int i = 0; i < inputsCount; i++)
                weights[i] = rnd.NextDouble();

            threshold = rnd.NextDouble();
        }

        /*
         * Misalnya, jaringan saraf untuk pengenalan tulisan tangan didefinisikan oleh satu set neuron masukan yang dapat diaktifkan dengan piksel dari gambar masukan. 
         * Setelah diberi bobot dan ditransofrmasikan dengan suatu fungsi aktifasi(ditentukan oleh desainer ANN), 
         * aktivasi dari neuron ini kemudian diteruskan ke neuron lain. Proses ini diulang sampai akhirnya, sebuah neuron output diaktifasi. 
         * 
         */
        public virtual double Compute(double[] input)
        {
            // cek apakah jumlah input sesuai dengan data input pada dataset
            if (input.Length != inputsCount)
                throw new ArgumentException("Wrong length of the input vector.");

            // inisialisasi fungsi penjumlahan
            double sum = 0.0;

            // hitung fungsi penjumlahan setiap bobot input
            for (int i = 0; i < weights.Length; i++)
            {
                sum += weights[i] * input[i];
            }
            // tambahkan hasil penjumlahan dengan threshold
            sum += threshold;

            // masukan fungsi aktifasi pada penjumlahan tadi kedalam variabel output
            double output = function.ActivationFunction(sum);
            // assign pada variabel output
            this.output = output;

            return output;
        }
    }
}
