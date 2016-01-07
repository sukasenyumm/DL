using System;
using LibDL.Interface;
using LibDL.Utils;

namespace LibDL
{
    [Serializable]
    public class StochasticNeuron : Neuron
    {
       
        private double outStochastic;
        public double OutStochastic
        {
            get { return outStochastic; }
        }

        
        public StochasticNeuron(int inputs, IActivationFunction function)
            : base(inputs, function)
        {

            // Ruslan Salakhutdinov dan Geoff Hinton memulai dengan nilai ambang batas 0
            this.Threshold = 0; 

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

            return output;
        }

        /*
         * From paper Estimating or Propagating Gradients Through Stochastic Neurons, Bengio Y, 2013
         * stochasticNeuron dihasilkan dari threshold ditambah dengan fungsi penjumlahan dari pembobotan input yang sudah diaktifasi
         * kemudian dikembalikan nilai dari fungsi aktifasi bernoulli atau gaussian yaitu 0 atau 1.
         * Dalam kasus ini digunakan bernoulli dengan asumsi bahwa data gambar sudah ternormalisasi pada setiap pixelnya bernilai 0 atau 1.
         */
        public double Generate(double[] input)
        {
            double sum = threshold;
            for (int i = 0; i < weights.Length; i++)
                sum += weights[i] * input[i];

            double output = function.ActivationFunction(sum);
            double stochasticNeuron = function.GenerateFromOutput(output);

            this.output = output;
            this.outStochastic = stochasticNeuron;

            return outStochastic;
        }

        /*
        * Fungsi tambahan jika neuron output sudah dihitung sebelumnya
        */
        public double Generate(double output)
        {
            double stochasticNeuron = function.GenerateFromOutput(output);

            this.output = output;
            this.outStochastic = stochasticNeuron;

            return stochasticNeuron;
        }
    }
}
