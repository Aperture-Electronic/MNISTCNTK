using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;
using static CNTK.CNTKLib;

namespace MNISTCNTK
{
    internal class MNISTConvolutionNN
    {
        private readonly Function cnn_function;
        private readonly Variable features;
        private readonly DeviceDescriptor device;

        public Function CNN_Function => cnn_function;

        private struct ConvolutionKernel
        {
            public int width, height;
            public int input_channel, output_map_channel;
            public int hStride, vStride;
            public int poolingWindowWidth, poolingWindowHeight;
        };

        public MNISTConvolutionNN(Variable features, DeviceDescriptor device, int outputDesmintations = 10, string classifierName = "classifierOutput")
        {
            // Set local private variables
            this.device = device;
            this.features = features;

            // Create and connect a convolution neutral network for MNIST
            // 1. Convolution layer I 
            // Kernel size = 3 x 3, Input size = 28 x 28 x 1, Output size = 12 x 12 x 30
            ConvolutionKernel kernel1 = new ConvolutionKernel()
            {
                width = 3,
                height = 3,
                input_channel = 1,
                output_map_channel = 30,
                vStride = 2,
                hStride = 2,
                poolingWindowWidth = 3,
                poolingWindowHeight = 3,
            };

            Function convolutionPooling1 = ConvolutionWithMaxPooling(inputLayer: features, kernel: kernel1);

            // 2. Convolution layer II
            // Kernel size = 3 x 3, Input size = 12 x 12 x 30, Output size =  8 x 8 x 10
            ConvolutionKernel kernel2 = new ConvolutionKernel()
            {
                width = 3,
                height = 3,
                input_channel = kernel1.output_map_channel,
                output_map_channel = 10,
                vStride = 1,
                hStride = 1,
                poolingWindowWidth = 3,
                poolingWindowHeight = 3,
            };

            Function convolutionPooling2 = ConvolutionWithMaxPooling(inputLayer: convolutionPooling1, kernel: kernel2);

            // 3. Reshape layer
            Function reshaped = Flatten(convolutionPooling2);
            // int planeDimension = ((Variable)convolutionPooling2).Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
            // Function reshaped = Reshape(convolutionPooling2, new int[] { planeDimension });

            // 4. Fully connected layer I 
            Function fullyConnectedI = Sigmoid(FullyConnectedLinearLayer(reshaped, 240, "outputl1"), "SigmoidI");

            // 5. Fully connected layer II 
            Function fullyConnectedII = Sigmoid(FullyConnectedLinearLayer(fullyConnectedI, 100, "outputl2"), "SigmoidII");

            // 6. Fully connected layer III
            Function fullyConnectedIII = FullyConnectedLinearLayer(fullyConnectedII, outputDesmintations, "outputl3");

            // 7. Softmax layer
            Function softmaxLayer = Softmax(fullyConnectedIII, classifierName);

            cnn_function = softmaxLayer;
        }

        private Function ConvolutionWithMaxPooling(Variable inputLayer, ConvolutionKernel kernel)
        {
            double convWScale = 0.26;

            // Configure a convolution layer
            Parameter conv_para = new Parameter(
                shape: new int[] { kernel.width, kernel.height, kernel.input_channel, kernel.output_map_channel },
                dataType: DataType.Float, initializer: GlorotUniformInitializer(convWScale, -1, 2), device: device);
            Function convolution = ReLU(Convolution(convolutionMap: conv_para, operand: inputLayer, strides: new int[] { 1, 1, kernel.input_channel }));

            Function pooling = Pooling(operand: convolution, poolingType: PoolingType.Max,
                poolingWindowShape: new int[] { kernel.poolingWindowWidth, kernel.poolingWindowHeight },
                strides: new int[] { kernel.hStride, kernel.vStride },
                autoPadding: new bool[] { true });

            return pooling;
        }

        private Function FullyConnectedLinearLayer(Variable input, int outputDimension, string outputName = "")
        {
            int inputDimension = input.Shape[0];

            int[] dimemsion = { outputDimension, inputDimension };
            Parameter timesParam = new Parameter(dimemsion, DataType.Float,
                GlorotUniformInitializer(DefaultParamInitScale, SentinelValueForInferParamInitRank, SentinelValueForInferParamInitRank, 1),
                device, "timesParam");
            Function timesFunction = Times(timesParam, input, "times");

            int[] output = { outputDimension };
            Parameter plusParam = new Parameter(output, 0.0f, device, "plusParam");
            return Plus(plusParam, timesFunction, outputName);
        }

        public static float ValidateModelWithMinibatchSource(string modelFile, MinibatchSource testMinibatchSource, int[] imageDim, int numClasses, string featureInputName, string labelInputName, string outputName, DeviceDescriptor device, int maxCount = 1000)
        {
            Function model = Function.Load(modelFile, device);
            Variable imageInput = model.Arguments[0];
            Variable labelOutput = model.Outputs.Single(o => o.Name == outputName);

            StreamInformation featureStreamInfo = testMinibatchSource.StreamInfo(featureInputName);
            StreamInformation labelStreamInfo = testMinibatchSource.StreamInfo(labelInputName);

            int batchSize = 50;
            int miscountTotal = 0, totalCount = 0;
            while (true)
            {
                UnorderedMapStreamInformationMinibatchData minibatchData = testMinibatchSource.GetNextMinibatch((uint)batchSize, device);
                if (minibatchData == null || minibatchData.Count == 0)
                {
                    break;
                }

                totalCount += (int)minibatchData[featureStreamInfo].numberOfSamples;

                // Expected labels are in the minibatch data.
                IList<IList<float>> labelData = minibatchData[labelStreamInfo].data.GetDenseData<float>(labelOutput);
                List<int> expectedLabels = labelData.Select(l => l.IndexOf(l.Max())).ToList();

                Dictionary<Variable, Value> inputDataMap = new Dictionary<Variable, Value>() {
                    { imageInput, minibatchData[featureStreamInfo].data }
                };

                Dictionary<Variable, Value> outputDataMap = new Dictionary<Variable, Value>() {
                    { labelOutput, null }
                };

                model.Evaluate(inputDataMap, outputDataMap, device);
                IList<IList<float>> outputData = outputDataMap[labelOutput].GetDenseData<float>(labelOutput);
                List<int> actualLabels = outputData.Select(l => l.IndexOf(l.Max())).ToList();

                int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();

                miscountTotal += misMatches;
                Console.WriteLine($"Validating Model: Total Samples = {totalCount}, Misclassify Count = {miscountTotal}");

                if (totalCount > maxCount)
                {
                    break;
                }
            }

            float errorRate = 1.0F * miscountTotal / totalCount;
            Console.WriteLine($"Model Validation Error = {errorRate}");
            return errorRate;
        }
    }
}
