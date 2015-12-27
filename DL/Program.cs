using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LibDL;
using LibDL.Utils;
using LibDL.Interface;
using LibDL.ActivationFunction;
using LibDL.Learning;
using Accord.Math;
using Accord.Statistics.Analysis;
using System.Diagnostics;
using System.IO;

using Accord.Neuro;
using Accord.Neuro.Networks;
using Accord.Neuro.Learning;
using AForge.Neuro.Learning;
using AForge.Neuro;
namespace DL
{
    class Program
    {
        public static TimeSpan procTime = new TimeSpan();
        static void Main(string[] args)
        {
            //var x = new double[100];
            //var y = new double[100];
            //for (int i = 0; i < 100; i++)
            //{
            //    x[i] = i;
            //    y[i] = 2 ^ i;
            //}
            //MyMatlab.MyGraph2D(x, y);
            //Console.Write("[any key to exit]");
            //Console.ReadKey();

            Console.WriteLine("Selamat Datang.."); 
            Console.WriteLine("Untuk mode penyimpanan ketik 'true' \ndan jika load file bin ketik 'false'(prefer)..");
            Console.Write("Mode: ");
            bool devMode = bool.Parse(Console.ReadLine());
            ReadCSVData data = new ReadCSVData();
            string file;
            string skip = "lala";
            string fileModel = "";
            int menu=1;
            
            if (devMode)
            {
                Console.WriteLine("Harap tunggu sebentar");
                data.LoadData(@"D:\train300.csv", true, true);
                data.Save(@"D:\train300sign.bin");
            }
            else
            {
                Console.WriteLine("Jika sudah punya file model skip proses learning dengan ketik 'skip'? ");
                skip = Console.ReadLine();
                if(skip == "skip")
                {
                    Console.WriteLine("Ketikkan direktori file model: ");
                    fileModel = @Console.ReadLine();
                }
             
                Console.WriteLine("Pilih menu berikut: \n1. Data MNIST TRAINING 784\n2. Data HURUF\n3. Data MNIST TRAINING 400");
                Console.Write("Pilih Menu: ");
                menu = int.Parse(Console.ReadLine());
                if (menu == 1)
                {
                    file = @"D:\data_mnist_train.bin";
                }
                else if (menu == 2)
                {
                    file = @"D:\dataHuruf.bin";
                }
                else if (menu == 3)
                {
                    file = @"D:\data_mnist_train.bin"; //for test just
                }
                else
                {
                    file = @"D:\data_mnist_train.bin";
                }
                data = ReadCSVData.Load(file);
                       
            }

            double[][] inputs;
            if (menu != 3)
                inputs = data.Inputs.ToArray();
            else
                inputs = data.WidthNormalization(data.Inputs.ToArray());
            double[][] outputs = data.Outputs.ToArray();

            string fileModelSimpan="";
            //DBN-rprop
            if (skip != "skip")
            {
                Console.Write("Simpan model dengan nama: ");
                fileModelSimpan = Console.ReadLine();
                TimeSpan begin = Process.GetCurrentProcess().TotalProcessorTime;
                List<double> errCostFunc = new List<double>();

                //RBM mc = new RBM(inputs.First().Length, 200);
                //RBM mc2 = new RBM(200, 200);
                //RBM mc3 = new RBM(200, outputs.First().Length);
                //DBN network = new DBN(inputs.First().Length, mc, mc2, mc3);

                //DBNLearning teacher = new DBNLearning(network)
                //{
                //    LearningRate = 0.1,
                //    Momentum = 0.5,
                //    WeightDecay = 0.001,
                //};
                //teacher.CreateCD();


                //// learn 5000 iterations
                //// Setup batches of input for learning.
                //int batchCount = Math.Max(1, inputs.Length / 100);
                //// Create mini-batches to speed learning.
                //int[] groups = Accord.Statistics.Tools.RandomGroups(inputs.Length, batchCount);
                //double[][][] batches = inputs.Subgroups(groups);
                //// Learning data for the specified layer.
                //double[][][] layerData;

                //var cd = teacher.GetLayerAlgorithm(teacher.LayerIndex);

                //// Unsupervised learning on each hidden layer, except for the output layer.
                //for (int layerIndex = 0; layerIndex < network.StackedRBM.Count - 1; layerIndex++)
                //{
                //    teacher.LayerIndex = layerIndex;
                //    layerData = teacher.GetLayerInput(batches);
                //    for (int i = 0; i < 200; i++)
                //    {
                //        double error = teacher.RunEpoch(layerData) / inputs.Length;
                //        errCostFunc.Add(error);
                //        if (i % 10 == 0)
                //        {
                //            Console.WriteLine(i + ", Error = " + error);

                //        }
                //        //if (i == 10)
                //        //    cd.Momentum = 0.9;
                //    }
                //}

                //network.UpdateVisibleWeights();

                //var teacher3 = new ResilientBackPropagationFineTuning(network);

                //for (int i = 0; i < 100; i++)
                //{
                //    double error = teacher3.RunEpoch(inputs, outputs) / inputs.Length;
                //    errCostFunc.Add(error);
                //    if (i % 10 == 0)
                //    {
                //        Console.WriteLine(i + ", Error = " + error);

                //    }
                //}

                //TimeSpan end = Process.GetCurrentProcess().TotalProcessorTime;

                RBM mc = new RBM(inputs.First().Length, 100);
                RBM mc2 = new RBM(100, 100);
                RBM mc21 = new RBM(100, 10);
                RBM mc3 = new RBM(10, outputs.First().Length);
                DBN network = new DBN(inputs.First().Length, mc, mc2, mc21, mc3);


                DBNLearning teacher = new DBNLearning(network)
                {
                    LearningRate = 0.1,
                    Momentum = 0.5,
                    WeightDecay = 0.001,
                };
                teacher.CreateCD();


                //learn 5000 iterations
                //Setup batches of input for learning.
                int batchCount = Math.Max(1, inputs.Length / 100);
                // Create mini-batches to speed learning.
                int[] groups = Accord.Statistics.Tools.RandomGroups(inputs.Length, batchCount);
                double[][][] batches = inputs.Subgroups(groups);
                // Learning data for the specified layer.
                double[][][] layerData;

                var cd = teacher.GetLayerAlgorithm(teacher.LayerIndex);

                // Unsupervised learning on each hidden layer, except for the output layer.
                for (int layerIndex = 0; layerIndex < network.StackedRBM.Count - 1; layerIndex++)
                {
                    teacher.LayerIndex = layerIndex;
                    layerData = teacher.GetLayerInput(batches);
                    for (int i = 0; i < 200; i++)
                    {
                        double error = teacher.RunEpoch(layerData) / inputs.Length;
                        errCostFunc.Add(error);
                        if (i % 10 == 0)
                        {
                            Console.WriteLine(i + ", Error = " + error);

                        }
                    }
                }

                network.UpdateVisibleWeights();
                network.Save(@fileModelSimpan);

                //DBN network = DBN.Load(@"D:\embuh");
                ////var teacher3 = new ResilientBackPropagationFineTuning(network);
                var teacher3 = new Backpropagation(network)
                {
                    LearningRate = 0.5,
                    Momentum = 0.05,
                };


                //learn 5000 iterations
                //Setup batches of input for learning.
                int batchCountBP = Math.Max(1, inputs.Length / 1000);
                // Create mini-batches to speed learning.
                int[] groupsBP = Random(inputs.Length, batchCountBP);
                double[][][] batchesIn = inputs.Subgroups(groupsBP);
                double[][][] batchesOut = outputs.Subgroups(groupsBP);
                // Learning data for the specified layer.
                double[][][] layerDataBPIn;
                double[][][] layerDataBPOut;
                layerDataBPIn = teacher3.GetLayerIO(batchesIn);
                layerDataBPOut = teacher3.GetLayerIO(batchesOut);


                for (int i = 0; i < 1000; i++)
                {
                    double error = teacher3.RunEpoch(inputs, outputs) / inputs.Length;
                    errCostFunc.Add(error);
                    if (i % 10 == 0)
                    {
                        Console.WriteLine(i + ", Error = " + error);

                    }
                }

                TimeSpan end = Process.GetCurrentProcess().TotalProcessorTime;

                //NeuralNetwork network = new NeuralNetwork(false,
                //    new LibDL.ActivationFunction.SigmoidFunction(0.001),
                //    inputs.First().Length, // two inputs in the network
                //    200, // two neurons in the first layer
                //    200,
                //    10); // one neuron in the second layer
                //// create teacher
                //var teacher3 = new BackpropTest(network)
                //{
                //    LearningRate = 0.5,
                //    Momentum = 0.5,
                //};// loop

                //for (int i = 0; i < 5000; i++)
                //{
                //    double error = teacher3.RunEpoch(inputs, outputs) / inputs.Length;
                //    errCostFunc.Add(error);
                //    if (i % 10 == 0)
                //    {
                //        Console.WriteLine(i + ", Error = " + error);

                //    }
                //}

                //TimeSpan end = Process.GetCurrentProcess().TotalProcessorTime;

                //network.setTimeLearn((end - begin).TotalMilliseconds + " ms.");
                //network.setAllErrCost(errCostFunc);

                //network.Save(@fileModelSimpan);
                //CreateLogValidation(network, inputs, outputs, data, false);//true untuk huruf

               // RestrictedBoltzmannMachine mc = new RestrictedBoltzmannMachine(inputs.First().Length, 100);
               //// RestrictedBoltzmannMachine mc2 = new RestrictedBoltzmannMachine(100, 100);
               // RestrictedBoltzmannMachine mc21 = new RestrictedBoltzmannMachine(100, 10);
               // RestrictedBoltzmannMachine mc3 = new RestrictedBoltzmannMachine(10, outputs.First().Length);
               // DeepBeliefNetwork network = new DeepBeliefNetwork(inputs.First().Length, mc, mc21, mc3);

               // DeepBeliefNetworkLearning teacher = new DeepBeliefNetworkLearning(network)
               // {
               //     Algorithm = (h, v, i) => new ContrastiveDivergenceLearning(h, v)
               //     {
               //         LearningRate = 0.5,
               //         Momentum = 0.5,
               //         Decay = 0.001,
               //     }
               // };
               // //teacher.CreateCD();


               // // learn 5000 iterations
               // // Setup batches of input for learning.
               // int batchCount = Math.Max(1, inputs.Length / 100);
               // // Create mini-batches to speed learning.
               // int[] groups = Accord.Statistics.Tools.RandomGroups(inputs.Length, batchCount);
               // double[][][] batches = inputs.Subgroups(groups);
               // // Learning data for the specified layer.
               // double[][][] layerData;

               // var cd = teacher.GetLayerAlgorithm(teacher.LayerIndex);

               // // Unsupervised learning on each hidden layer, except for the output layer.
               // for (int layerIndex = 0; layerIndex < network.Machines.Count - 1; layerIndex++)
               // {
               //     teacher.LayerIndex = layerIndex;
               //     layerData = teacher.GetLayerInput(batches);
               //     for (int i = 0; i < 200; i++)
               //     {
               //         double error = teacher.RunEpoch(layerData) / inputs.Length;
               //         errCostFunc.Add(error);
               //         if (i % 10 == 0)
               //         {
               //             Console.WriteLine(i + ", Error = " + error);

               //         }
               //     }
               // }

               // network.UpdateVisibleWeights();
               // //network.Save(@fileModelSimpan);
               // //DeepBeliefNetwork network = DeepBeliefNetwork.Load(@"D:\embuh");
               // //network.SetActivationFunction(new AForge.Neuro.(2));
               // //var teacher3 = new ResilientBackPropagationFineTuning(network);

               // var teacher3 = new BackPropagationLearning(network)
               // {
               //     LearningRate = 0.1,
               //     Momentum = 0.05,
               // };

                
               // for (int i = 0; i < 1000; i++)
               // {
               //     double error = teacher3.RunEpoch(inputs, outputs) / inputs.Length;
               //     errCostFunc.Add(error);
               //     if (i % 10 == 0)
               //     {
               //         Console.WriteLine(i + ", Error = " + error);

               //     }
               // }

                //TimeSpan end = Process.GetCurrentProcess().TotalProcessorTime;

                // create neural network
                //ActivationNetwork network = new ActivationNetwork(
                //    new AForge.Neuro.SigmoidFunction(0.001),
                //    inputs.First().Length, // two inputs in the network
                //    200, // two neurons in the first layer
                //    200,
                //    10); // one neuron in the second layer
                //// create teacher
                //var teacher3 = new BackPropagationLearning(network)
                //{
                //    LearningRate = 0.5,
                //    Momentum = 0.5,
                //};// loop

                //for (int i = 0; i < 5000; i++)
                //{
                //    double error = teacher3.RunEpoch(inputs, outputs) / inputs.Length;
                //    errCostFunc.Add(error);
                //    if (i % 10 == 0)
                //    {
                //        Console.WriteLine(i + ", Error = " + error);

                //    }
                //}

                
                //network.setTimeLearn((end - begin).TotalMilliseconds + " ms.");
                //network.setAllErrCost(errCostFunc);

                
                network.Save(@fileModelSimpan+"final");
                CreateLogValidation(network, inputs, outputs, data, false);

                //var psi = new ProcessStartInfo("shutdown", "/s /t 0");
                //psi.CreateNoWindow = true;
                //psi.UseShellExecute = false;
                //Process.Start(psi);
            }
            else
            {
                DBN network = DBN.Load(fileModel);
                CreateLogValidation(network, inputs, outputs, data, false);
            }
            //
            //
        }

        public static void CreateLogValidation(ActivationNetwork network, double[][] inputs, double[][] outputs, ReadCSVData data, bool isHuruf)
        {
            int[] actual = new int[outputs.Length];
            int[] expected = new int[outputs.Length];


            for (int i = 0; i < outputs.Length; i++)
            {
                actual[i] = data.ClassOutput(outputs[i], outputs.First().Length);
                if (isHuruf)
                {
                    if (actual[i] == 26)
                        actual[i] = 0;
                }
                else
                {
                    if (actual[i] == 10)
                        actual[i] = 0;
                }
            }

            int cou = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                expected[i] = data.ClassOutput(network.Compute(inputs[i]), outputs.First().Length);
                if (isHuruf)
                {
                    if (expected[i] == 26)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
                else
                {
                    if (expected[i] == 10)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
            }

            int numclassses = outputs.First().Length;
            if (isHuruf)
                numclassses = outputs.First().Length - 1;
            GeneralConfusionMatrix validation = new GeneralConfusionMatrix(numclassses, actual, expected);

            using (StreamWriter w = File.AppendText(@"D:\logAnalysis.txt"))
            {
                Log(validation, "debug", outputs, isHuruf, w);
                //Log("Test2", w);
            }

            using (StreamReader r = File.OpenText(@"D:\logAnalysis.txt"))
            {
                DumpLog(r);
            }
        }

        public static void CreateLogValidation(DBN network,double[][] inputs,double[][] outputs,ReadCSVData data,bool isHuruf)
        {
            int[] actual = new int[outputs.Length];
            int[] expected = new int[outputs.Length];


            for (int i = 0; i < outputs.Length; i++)
            {
                actual[i] = data.ClassOutput(outputs[i], outputs.First().Length);
                if (isHuruf)
                {
                    if (actual[i] == 26)
                        actual[i] = 0;
                }
                else
                {
                    if (actual[i] == 10)
                        actual[i] = 0;
                }
            }

            int cou = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                expected[i] = data.ClassOutput(network.ZeroOutput(network.Compute(inputs[i])), outputs.First().Length);
                if (isHuruf)
                {
                    if (expected[i] == 26)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
                else
                {
                    if (expected[i] == 10)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
            }

            int numclassses = outputs.First().Length;
            if (isHuruf)
                numclassses = outputs.First().Length - 1;
            GeneralConfusionMatrix validation = new GeneralConfusionMatrix(numclassses, actual, expected);

            using (StreamWriter w = File.AppendText(@"D:\logAnalysis.txt"))
            {
                Log(validation, network.getTimeLearn(), outputs, isHuruf, w);
                //Log("Test2", w);
            }

            using (StreamReader r = File.OpenText(@"D:\logAnalysis.txt"))
            {
                DumpLog(r);
            }
        }
        public static void CreateLogValidation(NeuralNetwork network, double[][] inputs, double[][] outputs, ReadCSVData data, bool isHuruf)
        {
            int[] actual = new int[outputs.Length];
            int[] expected = new int[outputs.Length];


            for (int i = 0; i < outputs.Length; i++)
            {
                actual[i] = data.ClassOutput(outputs[i], outputs.First().Length);
                if (isHuruf)
                {
                    if (actual[i] == 26)
                        actual[i] = 0;
                }
                else
                {
                    if (actual[i] == 10)
                        actual[i] = 0;
                }
            }

            int cou = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                expected[i] = data.ClassOutput(network.ZeroOutput(network.Compute(inputs[i])), outputs.First().Length);
                if (isHuruf)
                {
                    if (expected[i] == 26)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
                else
                {
                    if (expected[i] == 10)
                    {
                        cou++;
                        expected[i] = 0;
                    }
                }
            }

            int numclassses = outputs.First().Length;
            if (isHuruf)
                numclassses = outputs.First().Length - 1;
            GeneralConfusionMatrix validation = new GeneralConfusionMatrix(numclassses, actual, expected);

            using (StreamWriter w = File.AppendText(@"D:\logAnalysis.txt"))
            {
                Log(validation, network.getTimeLearn(), outputs, isHuruf, w);
                //Log("Test2", w);
            }

            using (StreamReader r = File.OpenText(@"D:\logAnalysis.txt"))
            {
                DumpLog(r);
            }
        }

        public static void Log(GeneralConfusionMatrix validation,string time,double[][] outputs, bool isHuruf, TextWriter w)
        {
            w.Write("\r\nLog Entry : ");
            w.WriteLine("{0} {1}", DateTime.Now.ToLongTimeString(),
                DateTime.Now.ToLongDateString());
            w.WriteLine("Learning time occured:" + time);
            w.WriteLine("============================================================================================");
            w.WriteLine("Number of Sample:" + validation.Samples);
            w.WriteLine("Accuracy:" + validation.OverallAgreement);
            w.WriteLine("Geometric Agreement:" + validation.GeometricAgreement);
            w.WriteLine("Chance Agreement:" + validation.ChanceAgreement);
            w.WriteLine("Kappa:" + validation.Kappa);
            w.WriteLine("Confusion Matrix:");

            int numOut = outputs.First().Length;
            if(isHuruf)
                numOut= outputs.First().Length-1;

            for (int j = 0; j < numOut; j++)
            {
                w.Write("{0,-5},", j);
            }
            w.WriteLine();
            w.WriteLine();
            for (int i = 0; i < numOut; i++)
            {
                for (int j = 0; j < numOut; j++)
                {
                    w.Write("{0,-5},", validation.Matrix[i, j]);
                }
                w.WriteLine();
            }
            w.WriteLine("============================================================================================");
        }

        public static void DumpLog(StreamReader r)
        {
            string line;
            while ((line = r.ReadLine()) != null)
            {
                Console.WriteLine(line);
            }
        }

        public static int[] Random(int size, int groups)
        {
            // Create the index vector
            int[] idx = new int[size];

            if (groups == 1)
            {
                for (int i = 0; i < idx.Length; i++)
                    idx[i] = 0;
                return idx;
            }

            double n = groups / (double)size;
            for (int i = 0; i < idx.Length; i++)
                idx[i] = (int)System.Math.Ceiling((i + 0.9) * n) - 1;

            // Shuffle the indices vector
            Helper.Shuffle(idx);

            return idx;
        }

        
    }
}
