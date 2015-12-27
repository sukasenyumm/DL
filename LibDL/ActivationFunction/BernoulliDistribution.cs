using System;
using LibDL.Interface;
using LibDL.Utils;

namespace LibDL.ActivationFunction
{
    [Serializable]
    public class BernoulliDistribution : IActivationFunction
    {
        /*
         * deklarasi alpha, alpha dapat membuat distribusi nilai fungsi aktifasi lebih halus dan dapat dikendalikan.
         * jika alpha diterapkan dalam algoritma learning (contoh.gradient decent) alpha merupakan parameter untuk learning rate.
         */
        private double alpha = 1;
        public double Alpha
        {
            get { return alpha; }
            set { alpha = value; }
        }
        /*
         * Generate nilai random
         */ 
        private static Random random = new ThreadSafeRandom();
        public static Random rnd { get { return random; } private set { random = value; } }
        /*
         * Konstruktor
         */
        public BernoulliDistribution()
        {
         
        }
        public BernoulliDistribution(double alpha)
        {
            this.alpha = alpha;
        }
        /*
         * logistic function digunakan dalam perhitungan estimasi probabilitas dimana hidden unit ke-j = 1, dengan visible unit x
         * dan parameternya adalah theta. Serta dalam perhitungan estimasi probabilitas visible unit ke-i = 1 dengan hidden unit h
         * dan parameternya adalah theta. logistic function pada neural network sering kali dikaitkan pada sigmoid function
         * pada NN.
         * https://en.wikipedia.org/wiki/Logistic_function
         * 
         */
        public double ActivationFunction(double x)
        {
            return (1 / (1 + Math.Exp(alpha * -x)));
        }
        /*
         * generate nilai boolean dari logistic function yang menghasilkan nilai distribusi bernoulli dari input x
         */
        public double GenerateFromInput(double x)
        {
            double y = ActivationFunction(x);
            return (y > rnd.NextDouble()) ? 1 : 0;
        }

        /*
         * generate nilai boolean dari logistic function yang menghasilkan nilai distribusi bernoulli dari output y, 
         * ekuivalen dengan fungsi GenerateFromInput
         */
        public double GenerateFromOutput(double y)
        {
            return (y > rnd.NextDouble()) ? 1 : 0;
        }
        /*
         * Nilai turunan derivatif dari distribusi bernoulli, https://en.wikipedia.org/wiki/Logistic_function, lihat pada
         * menu Logistic differential equation.
         * Derivative pada input x.
         * Derivative2 pada output y.
         *                 1
         * f(x) = ------------------
         *        1 + exp(-alpha * x)
         *
         *          alpha * exp(-alpha * x )
         * f'(x) = ---------------------------- = alpha * f(x) * (1 - f(x))
         *          (1 + exp(-alpha * x))^2
         */
        public double DerivativeInput(double x)
        {
            double y = ActivationFunction(x);
            return alpha * (y * (1 - y));
        }
        public double DerivativeOutput(double y)
        {
            return (alpha * y * (1 - y));
        }
    }
}
