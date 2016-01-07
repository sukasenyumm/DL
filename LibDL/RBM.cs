using System;
using LibDL.Interface;
using LibDL.Utils;
using LibDL.ActivationFunction;

namespace LibDL
{
    [Serializable]
    public class RBM : NeuralNetwork
    {
        /*
         * Seperti namanya, RBM adalah varian dari Boltzmann Machine, dengan batasan bahwa neuronnya harus memiliki bentuk
         * seperti graf bipatrid: sepasang node dari masing-masing dua kelompok unit, yang biasanya disebut dengan unit "visible" dan
         * unit "hidden". merekan mungkin memiliki koneksi yang simetris, dan tidak ada koneksi antar node dalam satu unit.
         * 
         * Sumber:https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine
         */

        /* 
         * Karena RMB merupakan model ANN stocastic, maka visible unit atau layer visible dibentuk dari stokastik layer atau unit.
         */ 
        private StochasticNetworkLayer visible;
        public StochasticNetworkLayer Visible
        {
            get { return visible; }
        }

        /* 
         * Karena RMB merupakan model ANN stocastic, maka hidden unit atau layer hidden dibentuk dari stokastik layer atau unit.
         */ 
        private StochasticNetworkLayer hidden;
        public StochasticNetworkLayer Hidden
        {
            get { return hidden; }
        }


         public RBM(int inputsCount, int hiddenNeurons)
            : this(new BernoulliDistribution(alpha: 1), inputsCount, hiddenNeurons) { }

        
         public RBM(StochasticNetworkLayer hidden, StochasticNetworkLayer visible)
             : base(null, hidden.InputsCount, 0)
        {
            this.hidden = hidden;
            this.visible = visible;

            base.layers[0] = hidden;
        }

        /* aktifkan RBM dengan membuat unit visible dan hidden beserta fungsi aktivasinya (selalu dimulai dengan bernoulli)
         * buat arsitektur jaringan syaraf dengan layer stokastik dan fungsi aktifasi bernoulli
         * bentuk layer visible dengan jumlah node sebanyak banyaknya input dari data
         * bentuk layer hidden dengan jumlah node yang disesuaikan tau banykanya node tertentu
         * gunakan arsitektur srokastik ann pada layer pertama sebagai hidden unit
         */ 
        public RBM(IActivationFunction function, int inputsCount, int hiddenNeurons)
             : base(function, inputsCount, 1)
        {
            this.visible = new StochasticNetworkLayer(inputsCount, hiddenNeurons,function);
            this.hidden = new StochasticNetworkLayer(hiddenNeurons, inputsCount, function);

            base.layers[0] = hidden;
        }

        /* keempat fungsi dibawah berguna untuk proses learning
         * 
         * Compute: fungs untuk 
         */ 
        public override double[] Compute(double[] input)
        {

            return hidden.Compute(input);
        }

        /*
         * 
         */ 
        public double[] Reconstruct(double[] output)
        {
            return visible.Compute(output);
        }

       
        public double[] GenerateOutput(double[] input)
        {
            return hidden.Generate(input);
        }

        public double[] GenerateInput(double[] output)
        {
            return visible.Generate(output);
        }

        public void UpdateVisibleWeights()
        {
            Visible.CopyReversedWeightsFrom(Hidden);
        }
    }
}
