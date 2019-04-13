using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;
using static CNTK.CNTKLib;
using static MNISTCNTK.MNISTData;

namespace MNISTCNTK
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            DeviceDescriptor device = DeviceDescriptor.CPUDevice;

            Console.WriteLine("Welcome to CNTK machine learning world");
            Console.WriteLine($"Device: {device.AsString()}");

            // Queue: Convert and create the CSV file for ML.net?
            Console.WriteLine("Do you want to convert and create the text minibatch file for CNTK? If the file is already exist, type 'N' to skip, else type 'Y' to avoid errors.");
            if (Console.ReadKey().Key == ConsoleKey.Y)
            {
                MNISTDataConvertor convertor = new MNISTDataConvertor();
                convertor.ConvertAndSave();
            }
            Console.WriteLine();

            string featureStreamName = "features";
            string labelsStreamName = "labels";
            string classifierName = "classifierOutput";
            int[] image_dimension = new int[] { 28, 28, 1 };
            int[] result_dimension = new int[] { 10 };
            int image_flat_size = 28 * 28;
            int number_class = 10;

            // Input and output
            Variable features = InputVariable(shape: image_dimension, dataType: DataType.Float, name: "features");
            Variable labels = InputVariable(shape: result_dimension, dataType: DataType.Float, name: "labels");

            // Scaled and CNN
            Function scaled_input = ElementTimes(Constant.Scalar(1.0f / 256.0f, device), features);
            MNISTConvolutionNN convolutionNN = new MNISTConvolutionNN(scaled_input, device, classifierName: classifierName);

            Function classifier_output = convolutionNN.CNN_Function;

            // Loss functions
            Function loss_function = CrossEntropyWithSoftmax(new Variable(classifier_output), labels, "lossFunction");
            Function perdiction = ClassificationError(new Variable(classifier_output), labels, "classificationError");

            // Learning rate
            TrainingParameterScheduleDouble trainingParameter = new TrainingParameterScheduleDouble(0.001125, 1);

            // Learners
            IList<Learner> learners = new List<Learner>()
            {
                Learner.SGDLearner(classifier_output.Parameters(), trainingParameter),
            };

            // Trainer
            Trainer trainer = Trainer.CreateTrainer(classifier_output, loss_function, perdiction, learners);

            // Minibatch(Training data)
            const uint minibatchSize = 50;
            int outputFrequencyMinibatch = 25;
            int Epchos = 8;
            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[]
            {
                new StreamConfiguration(featureStreamName, image_flat_size),
                new StreamConfiguration(labelsStreamName, number_class),
            };
            MinibatchSource minibatchSource = MinibatchSource.TextFormatMinibatchSource("mnist-train.txt", streamConfigurations, MinibatchSource.InfinitelyRepeat);
            StreamInformation featureStreamInfo = minibatchSource.StreamInfo(featureStreamName);
            StreamInformation labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName);

            // Training
            int iteration = 0;
            while (Epchos > 0)
            {
                UnorderedMapStreamInformationMinibatchData minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);
                Dictionary<Variable, MinibatchData> arguments = new Dictionary<Variable, MinibatchData>
                {
                    { features, minibatchData[featureStreamInfo] },
                    { labels, minibatchData[labelStreamInfo] }
                };

                trainer.TrainMinibatch(arguments, device);
                PrintTrainingProgress(trainer, iteration++, outputFrequencyMinibatch);

                if (minibatchData.Values.Any(a => a.sweepEnd))
                {
                    Epchos--;
                }
            }

            Console.WriteLine("*****Training done*****");

            // Save the trained model
            classifier_output.Save("cnn-model.mld");

            // Validate the model
            MinibatchSource minibatchSourceNewModel = MinibatchSource.TextFormatMinibatchSource("mnist-train.txt", streamConfigurations, MinibatchSource.FullDataSweep);
            MNISTConvolutionNN.ValidateModelWithMinibatchSource("cnn-model.mld", minibatchSourceNewModel,
                                image_dimension, number_class, featureStreamName, labelsStreamName, classifierName, device);

            Console.ReadKey(true);
        }

        public static void PrintTrainingProgress(Trainer trainer, int minibatchIdx, int outputFrequencyInMinibatches)
        {
            if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
            {
                float trainLossValue = (float)trainer.PreviousMinibatchLossAverage();
                float evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage();
                uint sampleCount = trainer.PreviousMinibatchSampleCount();
                Console.WriteLine($"[Minibatch {minibatchIdx}] Cross Entropy Loss: {trainLossValue}, Evaluation Criterion: {evaluationValue}, Sample Count: {sampleCount}");
            }
        }
    }
}
