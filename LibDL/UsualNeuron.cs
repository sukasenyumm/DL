using System;
using LibDL.Interface;
using LibDL.Utils;

namespace LibDL
{
    [Serializable]
    public class UsualNeuron : Neuron
    {

        private double outUsual;
        public double OutUsual
        {
            get { return outUsual; }
        }

        /*
         * Deklarasi fungsi aktifasi
         */
        private new IActivationFunction function;
        public IActivationFunction ActivationFunction
        {
            get { return function; }
            set { base.function = this.function = value; }
        }

        /*
         * Inisialiasi neuron dari input yang masuk
         */
        protected double threshold = 0.0;
        public double Threshold
        {
            get { return threshold; }
            set { threshold = value; }
        }
        public UsualNeuron(int inputs, IActivationFunction function)
            : base(inputs,function)
        {

            //inisialisasi fungsi aktivasi
            this.function = function;
        }

        /*
         * Misalnya, jaringan saraf untuk pengenalan tulisan tangan didefinisikan oleh satu set neuron masukan yang dapat diaktifkan dengan piksel dari gambar masukan. 
         * Setelah diberi bobot dan ditransofrmasikan dengan suatu fungsi aktifasi(ditentukan oleh desainer ANN), 
         * aktivasi dari neuron ini kemudian diteruskan ke neuron lain. Proses ini diulang sampai akhirnya, sebuah neuron output diaktifasi. 
         * 
         */
        public override double Compute(double[] input)
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
            this.outUsual = output;

            return outUsual;
        }
        public override void RandomWeights()
        {
            // randomize weights
            base.RandomWeights();
            for (int i = 0; i < inputsCount; i++)
                weights[i] = rnd.NextDouble();

            threshold = rnd.NextDouble();
        }
    }
}
