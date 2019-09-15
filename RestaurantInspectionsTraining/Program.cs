using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.AutoML;
using RestaurantInspectionsML;

namespace RestaurantInspectionsTraining
{
    class Program
    {
        static void Main(string[] args)
        {
            // Define source data directory paths
            string solutionDirectory = "/home/lqdev/Development/RestaurantInspectionsSparkMLNET";
            string dataLocation = Path.Combine(solutionDirectory,"RestaurantInspectionsETL","Output");

            // Initialize MLContext
            MLContext mlContext = new MLContext();

            // Get directory name of most recent ETL output
            var latestOutput = 
                Directory
                    .GetDirectories(dataLocation)
                    .Select(directory => new DirectoryInfo(directory))
                    .OrderBy(directoryInfo => directoryInfo.Name)
                    .Select(directory => Path.Join(directory.FullName,"Graded"))
                    .First();
            
            var dataFilePaths = 
                Directory
                    .GetFiles(latestOutput)
                    .Where(file => file.EndsWith("csv"))
                    .ToArray();

            // Load the data
            var dataLoader = mlContext.Data.CreateTextLoader<ModelInput>(separatorChar:',', hasHeader:false, allowQuoting:true, trimWhitespace:true);
            IDataView data = dataLoader.Load(dataFilePaths);

            // Split the data
            TrainTestData dataSplit = mlContext.Data.TrainTestSplit(data,testFraction:0.2);
            IDataView trainData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;

            // Define experiment settings
            var experimentSettings = new MulticlassExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 600;
            experimentSettings.OptimizingMetric = MulticlassClassificationMetric.LogLoss;

            // Create experiment
            var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(experimentSettings);

            // Run experiment
            var experimentResults = experiment.Execute(data, progressHandler: new ProgressHandler());

            // Best Run Results
            var bestModel = experimentResults.BestRun.Model;

            // Evaluate Model
            IDataView scoredTestData = bestModel.Transform(testData);  
            var metrics = mlContext.MulticlassClassification.Evaluate(scoredTestData);
            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}");

            // Save Model
            string modelSavePath = Path.Join(solutionDirectory,"RestaurantInspectionsML","model.zip");
            mlContext.Model.Save(bestModel, data.Schema, modelSavePath);
        }
    }
}
